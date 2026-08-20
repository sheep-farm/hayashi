use crate::lang::error::{HayashiError, Result};
use crate::lang::predicate::{GroupBounds, RowAccess, RowPredicate};
use arrow::array::{self, Array, AsArray, BooleanArray};
use arrow::datatypes::DataType as ArrowType;
use arrow::record_batch::RecordBatch;
use arrow::temporal_conversions::{
    date32_to_datetime, date64_to_datetime, timestamp_ms_to_datetime, timestamp_ns_to_datetime,
    timestamp_s_to_datetime, timestamp_us_to_datetime,
};
use greeners::DataFrame;
use parquet::arrow::arrow_reader::{ArrowPredicateFn, ParquetRecordBatchReaderBuilder, RowFilter};
use parquet::arrow::{ArrowWriter, ProjectionMask};
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use parquet::file::statistics::Statistics;
use std::collections::HashMap;
use std::fs::File;
use std::sync::Arc;

pub fn load_parquet(
    path: &str,
    columns: Option<&[String]>,
    predicate: Option<&RowPredicate>,
) -> Result<(DataFrame, usize)> {
    let file = File::open(path)
        .map_err(|e| HayashiError::Runtime(format!("cannot open '{path}': {e}")))?;

    let mut builder_reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| HayashiError::Runtime(format!("parquet error: {e}")))?;

    let schema_desc = builder_reader.parquet_schema().clone();
    let arrow_schema = builder_reader.schema().clone();

    // ── Projeção: colunas pedidas pelo usuário (ou todas). ───────────────
    let projection_cols: Vec<String> = match columns {
        Some(cols) if !cols.is_empty() => cols.to_vec(),
        _ => arrow_schema
            .fields()
            .iter()
            .map(|f| f.name().to_string())
            .collect(),
    };

    // Validar que as colunas pedidas existem no schema.
    let avail: std::collections::HashSet<String> = arrow_schema
        .fields()
        .iter()
        .map(|f| f.name().to_string())
        .collect();
    for c in &projection_cols {
        if !avail.contains(c) {
            return Err(HayashiError::Runtime(format!(
                "load parquet: column '{c}' not found — available: {}",
                arrow_schema
                    .fields()
                    .iter()
                    .map(|f| f.name().as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
    }

    let projection_mask =
        ProjectionMask::columns(&schema_desc, projection_cols.iter().map(|s| s.as_str()));

    // ── Filtro: where= via RowFilter do parquet (pushdown ao row group). ─
    if let Some(pred) = predicate {
        // Validar que todas as colunas referenciadas pelo predicado existem.
        for c in pred.referenced_columns() {
            if !avail.contains(&c) {
                return Err(HayashiError::Runtime(format!(
                    "load parquet: where references unknown column '{c}'"
                )));
            }
        }

        // ── Row group pruning por estatísticas (min/max). ───────────────
        // Antes de aplicar o RowFilter (que decodifica as colunas do
        // predicado), lemos os metadados de cada row group e pulamos os
        // groups onde o predicado não pode casar. Isso evita ler e
        // decodificar row groups inteiros.
        let parquet_metadata = builder_reader.metadata().clone();
        let row_groups = parquet_metadata.row_groups();
        let total_rgs = row_groups.len();

        // Mapear nome da coluna → índice no schema do parquet.
        let parquet_schema = parquet_metadata.file_metadata().schema_descr();
        let col_name_to_idx: HashMap<String, usize> = parquet_schema
            .columns()
            .iter()
            .enumerate()
            .map(|(i, c)| (c.name().to_string(), i))
            .collect();

        let mut kept_rgs: Vec<usize> = Vec::with_capacity(total_rgs);
        let mut pruned_count: usize = 0;
        for (rg_idx, rg) in row_groups.iter().enumerate() {
            let bounds = RowGroupStatsLookup {
                rg,
                col_name_to_idx: &col_name_to_idx,
            };
            if pred.can_match(&bounds) {
                kept_rgs.push(rg_idx);
            } else {
                pruned_count += 1;
            }
        }

        if pruned_count > 0 {
            builder_reader = builder_reader.with_row_groups(kept_rgs);
        }

        // A máscara do predicado inclui só as colunas que ele precisa.
        let pred_cols = pred.referenced_columns();
        let pred_mask = ProjectionMask::columns(&schema_desc, pred_cols.iter().map(|s| s.as_str()));
        let pred_clone = pred.clone();
        let arrow_pred = ArrowPredicateFn::new(pred_mask, move |batch: RecordBatch| {
            let n = batch.num_rows();
            let schema = batch.schema();
            let col_idx: HashMap<String, usize> = schema
                .fields()
                .iter()
                .enumerate()
                .map(|(i, f)| (f.name().to_string(), i))
                .collect();
            let mut bools = Vec::with_capacity(n);
            for i in 0..n {
                let row = ArrowRow {
                    batch: &batch,
                    idx: i,
                    col_idx: &col_idx,
                };
                bools.push(pred_clone.evaluate(&row));
            }
            Ok(BooleanArray::from(bools))
        });
        builder_reader = builder_reader.with_row_filter(RowFilter::new(vec![Box::new(arrow_pred)]));
    }

    builder_reader = builder_reader.with_projection(projection_mask);

    let reader = builder_reader
        .build()
        .map_err(|e| HayashiError::Runtime(format!("parquet reader error: {e}")))?;

    // ── Acumulação: mesmas regras de conversão do loader original. ───────
    // A coluna i do batch corresponde à coluna i de projection_cols, porque
    // o parquet devolve apenas as colunas projetadas, na ordem informada.
    let mut col_data: Vec<(String, ColAccum)> = Vec::new();
    let mut initialized = false;
    let mut n_rows: usize = 0;

    for batch_result in reader {
        let batch =
            batch_result.map_err(|e| HayashiError::Runtime(format!("parquet batch error: {e}")))?;

        if !initialized {
            // Mapear nome → índice no batch projetado (não no schema original).
            let batch_schema = batch.schema();
            for field in batch_schema.fields().iter() {
                let name = field.name().clone();
                let is_num = matches!(
                    field.data_type(),
                    ArrowType::Float16
                        | ArrowType::Float32
                        | ArrowType::Float64
                        | ArrowType::Int8
                        | ArrowType::Int16
                        | ArrowType::Int32
                        | ArrowType::Int64
                        | ArrowType::UInt8
                        | ArrowType::UInt16
                        | ArrowType::UInt32
                        | ArrowType::UInt64
                        | ArrowType::Boolean
                );
                // Para colunas que o usuário pediu mas não são numéricas,
                // acumular como String (timestamp/caem no braço de strings).
                col_data.push((
                    name,
                    if is_num {
                        ColAccum::Floats(Vec::new())
                    } else {
                        ColAccum::Strings(Vec::new())
                    },
                ));
            }
            initialized = true;
        }

        let rows_in_batch = batch.num_rows();
        n_rows += rows_in_batch;

        for (col_idx, (_name, accum)) in col_data.iter_mut().enumerate() {
            let col = batch.column(col_idx);
            match accum {
                ColAccum::Floats(ref mut vals) => {
                    append_as_f64(col, vals);
                }
                ColAccum::Strings(ref mut vals) => {
                    append_as_string(col, vals);
                }
            }
        }
    }

    let mut df_builder = DataFrame::builder();
    for (name, accum) in col_data {
        match accum {
            ColAccum::Floats(vals) => {
                df_builder = df_builder.add_column(&name, vals);
            }
            ColAccum::Strings(vals) => {
                df_builder = df_builder.add_string(&name, vals);
            }
        }
    }

    let df = df_builder
        .build()
        .map_err(|e| HayashiError::Runtime(format!("DataFrame build error: {e}")))?;

    Ok((df, n_rows))
}

pub fn write_parquet(df: &DataFrame, path: &str) -> Result<()> {
    use arrow::datatypes::{Field, Schema};

    let col_names = df.column_names();
    let n_rows = df.n_rows();

    let mut fields = Vec::new();
    let mut arrays: Vec<Arc<dyn Array>> = Vec::new();

    for name in &col_names {
        match df.get_column(name) {
            Ok(greeners::Column::Float(arr)) => {
                fields.push(Field::new(name, ArrowType::Float64, true));
                let values: Vec<f64> = arr.iter().copied().collect();
                arrays.push(Arc::new(array::Float64Array::from(values)));
            }
            Ok(greeners::Column::Int(arr)) => {
                fields.push(Field::new(name, ArrowType::Int64, true));
                let values: Vec<i64> = arr.iter().copied().collect();
                arrays.push(Arc::new(array::Int64Array::from(values)));
            }
            Ok(greeners::Column::Bool(arr)) => {
                fields.push(Field::new(name, ArrowType::Boolean, true));
                let values: Vec<bool> = arr.iter().copied().collect();
                arrays.push(Arc::new(array::BooleanArray::from(values)));
            }
            Ok(greeners::Column::String(arr)) => {
                fields.push(Field::new(name, ArrowType::Utf8, true));
                let values: Vec<&str> = arr.iter().map(|s| s.as_str()).collect();
                arrays.push(Arc::new(array::StringArray::from(values)));
            }
            Ok(greeners::Column::Categorical(cat)) => {
                fields.push(Field::new(name, ArrowType::Utf8, true));
                let values: Vec<String> = (0..n_rows)
                    .map(|i| cat.get_string(i).unwrap_or("").to_string())
                    .collect();
                let refs: Vec<&str> = values.iter().map(|s| s.as_str()).collect();
                arrays.push(Arc::new(array::StringArray::from(refs)));
            }
            _ => {
                fields.push(Field::new(name, ArrowType::Utf8, true));
                let empty: Vec<&str> = vec![""; n_rows];
                arrays.push(Arc::new(array::StringArray::from(empty)));
            }
        }
    }

    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), arrays)
        .map_err(|e| HayashiError::Runtime(format!("arrow batch error: {e}")))?;

    let file = File::create(path)
        .map_err(|e| HayashiError::Runtime(format!("cannot create '{path}': {e}")))?;

    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();

    let mut writer = ArrowWriter::try_new(file, schema, Some(props))
        .map_err(|e| HayashiError::Runtime(format!("parquet writer error: {e}")))?;

    writer
        .write(&batch)
        .map_err(|e| HayashiError::Runtime(format!("parquet write error: {e}")))?;

    writer
        .close()
        .map_err(|e| HayashiError::Runtime(format!("parquet close error: {e}")))?;

    Ok(())
}

enum ColAccum {
    Floats(Vec<f64>),
    Strings(Vec<String>),
}

fn append_as_f64(col: &dyn Array, out: &mut Vec<f64>) {
    match col.data_type() {
        ArrowType::Float64 => {
            let arr = col.as_primitive::<arrow::datatypes::Float64Type>();
            for i in 0..arr.len() {
                out.push(if arr.is_null(i) {
                    f64::NAN
                } else {
                    arr.value(i)
                });
            }
        }
        ArrowType::Float32 => {
            let arr = col.as_primitive::<arrow::datatypes::Float32Type>();
            for i in 0..arr.len() {
                out.push(if arr.is_null(i) {
                    f64::NAN
                } else {
                    arr.value(i) as f64
                });
            }
        }
        ArrowType::Int64 => {
            let arr = col.as_primitive::<arrow::datatypes::Int64Type>();
            for i in 0..arr.len() {
                out.push(if arr.is_null(i) {
                    f64::NAN
                } else {
                    arr.value(i) as f64
                });
            }
        }
        ArrowType::Int32 => {
            let arr = col.as_primitive::<arrow::datatypes::Int32Type>();
            for i in 0..arr.len() {
                out.push(if arr.is_null(i) {
                    f64::NAN
                } else {
                    arr.value(i) as f64
                });
            }
        }
        ArrowType::Int16 => {
            let arr = col.as_primitive::<arrow::datatypes::Int16Type>();
            for i in 0..arr.len() {
                out.push(if arr.is_null(i) {
                    f64::NAN
                } else {
                    arr.value(i) as f64
                });
            }
        }
        ArrowType::Int8 => {
            let arr = col.as_primitive::<arrow::datatypes::Int8Type>();
            for i in 0..arr.len() {
                out.push(if arr.is_null(i) {
                    f64::NAN
                } else {
                    arr.value(i) as f64
                });
            }
        }
        ArrowType::UInt64 => {
            let arr = col.as_primitive::<arrow::datatypes::UInt64Type>();
            for i in 0..arr.len() {
                out.push(if arr.is_null(i) {
                    f64::NAN
                } else {
                    arr.value(i) as f64
                });
            }
        }
        ArrowType::UInt32 => {
            let arr = col.as_primitive::<arrow::datatypes::UInt32Type>();
            for i in 0..arr.len() {
                out.push(if arr.is_null(i) {
                    f64::NAN
                } else {
                    arr.value(i) as f64
                });
            }
        }
        ArrowType::Boolean => {
            let arr = col.as_boolean();
            for i in 0..arr.len() {
                out.push(if arr.is_null(i) {
                    f64::NAN
                } else if arr.value(i) {
                    1.0
                } else {
                    0.0
                });
            }
        }
        _ => {
            for _ in 0..col.len() {
                out.push(f64::NAN);
            }
        }
    }
}

fn append_as_string(col: &dyn Array, out: &mut Vec<String>) {
    match col.data_type() {
        ArrowType::Utf8 => {
            let arr = col.as_string::<i32>();
            for i in 0..arr.len() {
                out.push(if arr.is_null(i) {
                    String::new()
                } else {
                    arr.value(i).to_string()
                });
            }
        }
        ArrowType::LargeUtf8 => {
            let arr = col.as_string::<i64>();
            for i in 0..arr.len() {
                out.push(if arr.is_null(i) {
                    String::new()
                } else {
                    arr.value(i).to_string()
                });
            }
        }
        // Timestamps: formatar como ISO 8601 (YYYY-MM-DD ou YYYY-MM-DDTHH:MM:SS).
        // O Hayashi não tem tipo Date nativo, então a melhor representação para
        // econometria é string ISO — o usuário pode fazer substr() para extrair
        // ano/mês/dia, ou generate para converter em numérico (dias desde epoch).
        ArrowType::Timestamp(arrow::datatypes::TimeUnit::Second, _) => {
            let arr = col.as_primitive::<arrow::datatypes::TimestampSecondType>();
            for i in 0..arr.len() {
                out.push(if arr.is_null(i) {
                    String::new()
                } else {
                    timestamp_s_to_datetime(arr.value(i))
                        .map(|dt| format_iso(&dt))
                        .unwrap_or_default()
                });
            }
        }
        ArrowType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, _) => {
            let arr = col.as_primitive::<arrow::datatypes::TimestampMillisecondType>();
            for i in 0..arr.len() {
                out.push(if arr.is_null(i) {
                    String::new()
                } else {
                    timestamp_ms_to_datetime(arr.value(i))
                        .map(|dt| format_iso(&dt))
                        .unwrap_or_default()
                });
            }
        }
        ArrowType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, _) => {
            let arr = col.as_primitive::<arrow::datatypes::TimestampMicrosecondType>();
            for i in 0..arr.len() {
                out.push(if arr.is_null(i) {
                    String::new()
                } else {
                    timestamp_us_to_datetime(arr.value(i))
                        .map(|dt| format_iso(&dt))
                        .unwrap_or_default()
                });
            }
        }
        ArrowType::Timestamp(arrow::datatypes::TimeUnit::Nanosecond, _) => {
            let arr = col.as_primitive::<arrow::datatypes::TimestampNanosecondType>();
            for i in 0..arr.len() {
                out.push(if arr.is_null(i) {
                    String::new()
                } else {
                    timestamp_ns_to_datetime(arr.value(i))
                        .map(|dt| format_iso(&dt))
                        .unwrap_or_default()
                });
            }
        }
        ArrowType::Date32 => {
            let arr = col.as_primitive::<arrow::datatypes::Date32Type>();
            for i in 0..arr.len() {
                out.push(if arr.is_null(i) {
                    String::new()
                } else {
                    date32_to_datetime(arr.value(i))
                        .map(|dt| format_iso(&dt))
                        .unwrap_or_default()
                });
            }
        }
        ArrowType::Date64 => {
            let arr = col.as_primitive::<arrow::datatypes::Date64Type>();
            for i in 0..arr.len() {
                out.push(if arr.is_null(i) {
                    String::new()
                } else {
                    date64_to_datetime(arr.value(i))
                        .map(|dt| format_iso(&dt))
                        .unwrap_or_default()
                });
            }
        }
        _ => {
            for i in 0..col.len() {
                out.push(if col.is_null(i) {
                    String::new()
                } else {
                    format!("{:?}", col)
                });
            }
        }
    }
}

/// Formata um `NaiveDateTime` como ISO 8601. Se a parte de tempo for meia-noite
/// (00:00:00), retorna apenas a data (YYYY-MM-DD); caso contrário, retorna
/// data e hora (YYYY-MM-DDTHH:MM:SS).
fn format_iso(dt: &chrono::NaiveDateTime) -> String {
    if dt.time() == chrono::NaiveTime::default() {
        dt.format("%Y-%m-%d").to_string()
    } else {
        dt.format("%Y-%m-%dT%H:%M:%S").to_string()
    }
}

// ── Suporte ao where= via RowFilter do parquet ─────────────────────────────

/// Wrapper sobre um row group do Parquet para implementar `GroupBounds`.
/// Permite avaliar `RowPredicate::can_match` contra as estatísticas
/// (min/max) de cada coluna sem decodificar os dados.
struct RowGroupStatsLookup<'a> {
    rg: &'a parquet::file::metadata::RowGroupMetaData,
    col_name_to_idx: &'a HashMap<String, usize>,
}

impl<'a> GroupBounds for RowGroupStatsLookup<'a> {
    fn f64_bounds(&self, col: &str) -> Option<(f64, f64)> {
        let idx = *self.col_name_to_idx.get(col)?;
        let col_chunk = self.rg.column(idx);
        let stats = col_chunk.statistics()?;
        match stats {
            Statistics::Int32(s) => {
                let min = (*s.min_opt()?) as f64;
                let max = (*s.max_opt()?) as f64;
                Some((min, max))
            }
            Statistics::Int64(s) => {
                let min = (*s.min_opt()?) as f64;
                let max = (*s.max_opt()?) as f64;
                Some((min, max))
            }
            Statistics::Int96(_) => None, // raro, conservador
            Statistics::Float(s) => {
                let min = (*s.min_opt()?) as f64;
                let max = (*s.max_opt()?) as f64;
                Some((min, max))
            }
            Statistics::Double(s) => {
                let min = *s.min_opt()?;
                let max = *s.max_opt()?;
                Some((min, max))
            }
            Statistics::Boolean(s) => {
                let min = if *s.min_opt()? { 1.0 } else { 0.0 };
                let max = if *s.max_opt()? { 1.0 } else { 0.0 };
                Some((min, max))
            }
            _ => None,
        }
    }

    fn str_bounds(&self, col: &str) -> Option<(&str, &str)> {
        let idx = *self.col_name_to_idx.get(col)?;
        let col_chunk = self.rg.column(idx);
        let stats = col_chunk.statistics()?;
        match stats {
            Statistics::ByteArray(s) => {
                let min_bytes = s.min_bytes_opt()?;
                let max_bytes = s.max_bytes_opt()?;
                let min = std::str::from_utf8(min_bytes).ok()?;
                let max = std::str::from_utf8(max_bytes).ok()?;
                Some((min, max))
            }
            Statistics::FixedLenByteArray(s) => {
                let min_bytes = s.min_bytes_opt()?;
                let max_bytes = s.max_bytes_opt()?;
                let min = std::str::from_utf8(min_bytes).ok()?;
                let max = std::str::from_utf8(max_bytes).ok()?;
                Some((min, max))
            }
            _ => None,
        }
    }
}

/// Linha de um `RecordBatch` projetada para avaliação do predicado `where`.
struct ArrowRow<'a> {
    batch: &'a RecordBatch,
    idx: usize,
    col_idx: &'a HashMap<String, usize>,
}

impl<'a> RowAccess for ArrowRow<'a> {
    fn get_f64(&self, col: &str) -> Option<f64> {
        let i = *self.col_idx.get(col)?;
        arrow_array_to_f64(self.batch.column(i).as_ref(), self.idx)
    }

    fn get_str(&self, col: &str) -> Option<&str> {
        let i = *self.col_idx.get(col)?;
        let arr = self.batch.column(i);
        match arr.data_type() {
            ArrowType::Utf8 => {
                let a = arr.as_string::<i32>();
                Some(if a.is_null(self.idx) {
                    ""
                } else {
                    a.value(self.idx)
                })
            }
            ArrowType::LargeUtf8 => {
                let a = arr.as_string::<i64>();
                Some(if a.is_null(self.idx) {
                    ""
                } else {
                    a.value(self.idx)
                })
            }
            _ => None,
        }
    }
}

fn arrow_array_to_f64(arr: &dyn Array, idx: usize) -> Option<f64> {
    use arrow::datatypes as dt;
    match arr.data_type() {
        ArrowType::Float64 => {
            let a = arr.as_primitive::<dt::Float64Type>();
            Some(if a.is_null(idx) {
                f64::NAN
            } else {
                a.value(idx)
            })
        }
        ArrowType::Float32 => {
            let a = arr.as_primitive::<dt::Float32Type>();
            Some(if a.is_null(idx) {
                f64::NAN
            } else {
                a.value(idx) as f64
            })
        }
        ArrowType::Int64 => {
            let a = arr.as_primitive::<dt::Int64Type>();
            Some(if a.is_null(idx) {
                f64::NAN
            } else {
                a.value(idx) as f64
            })
        }
        ArrowType::Int32 => {
            let a = arr.as_primitive::<dt::Int32Type>();
            Some(if a.is_null(idx) {
                f64::NAN
            } else {
                a.value(idx) as f64
            })
        }
        ArrowType::Int16 => {
            let a = arr.as_primitive::<dt::Int16Type>();
            Some(if a.is_null(idx) {
                f64::NAN
            } else {
                a.value(idx) as f64
            })
        }
        ArrowType::Int8 => {
            let a = arr.as_primitive::<dt::Int8Type>();
            Some(if a.is_null(idx) {
                f64::NAN
            } else {
                a.value(idx) as f64
            })
        }
        ArrowType::UInt64 => {
            let a = arr.as_primitive::<dt::UInt64Type>();
            Some(if a.is_null(idx) {
                f64::NAN
            } else {
                a.value(idx) as f64
            })
        }
        ArrowType::UInt32 => {
            let a = arr.as_primitive::<dt::UInt32Type>();
            Some(if a.is_null(idx) {
                f64::NAN
            } else {
                a.value(idx) as f64
            })
        }
        ArrowType::UInt16 => {
            let a = arr.as_primitive::<dt::UInt16Type>();
            Some(if a.is_null(idx) {
                f64::NAN
            } else {
                a.value(idx) as f64
            })
        }
        ArrowType::UInt8 => {
            let a = arr.as_primitive::<dt::UInt8Type>();
            Some(if a.is_null(idx) {
                f64::NAN
            } else {
                a.value(idx) as f64
            })
        }
        ArrowType::Boolean => {
            let a = arr.as_boolean();
            Some(if a.is_null(idx) {
                f64::NAN
            } else if a.value(idx) {
                1.0
            } else {
                0.0
            })
        }
        ArrowType::Utf8 => {
            let a = arr.as_string::<i32>();
            if a.is_null(idx) {
                Some(f64::NAN)
            } else {
                Some(a.value(idx).parse::<f64>().unwrap_or(f64::NAN))
            }
        }
        ArrowType::LargeUtf8 => {
            let a = arr.as_string::<i64>();
            if a.is_null(idx) {
                Some(f64::NAN)
            } else {
                Some(a.value(idx).parse::<f64>().unwrap_or(f64::NAN))
            }
        }
        _ => None,
    }
}
