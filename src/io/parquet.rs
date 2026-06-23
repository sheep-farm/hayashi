use crate::lang::error::{HayashiError, Result};
use arrow::array::{self, Array, AsArray};
use arrow::datatypes::DataType as ArrowType;
use greeners::DataFrame;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::sync::Arc;

pub fn load_parquet(path: &str) -> Result<(DataFrame, usize)> {
    let file = File::open(path)
        .map_err(|e| HayashiError::Runtime(format!("cannot open '{path}': {e}")))?;

    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| HayashiError::Runtime(format!("parquet error: {e}")))?
        .build()
        .map_err(|e| HayashiError::Runtime(format!("parquet reader error: {e}")))?;

    let mut builder = DataFrame::builder();
    let mut n_rows: usize = 0;
    let mut col_data: Vec<(String, ColAccum)> = Vec::new();
    let mut initialized = false;

    for batch_result in reader {
        let batch =
            batch_result.map_err(|e| HayashiError::Runtime(format!("parquet batch error: {e}")))?;

        let schema = batch.schema();
        if !initialized {
            for field in schema.fields() {
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

    for (name, accum) in col_data {
        match accum {
            ColAccum::Floats(vals) => {
                builder = builder.add_column(&name, vals);
            }
            ColAccum::Strings(vals) => {
                builder = builder.add_string(&name, vals);
            }
        }
    }

    let df = builder
        .build()
        .map_err(|e| HayashiError::Runtime(format!("DataFrame build error: {e}")))?;

    Ok((df, n_rows))
}

pub fn write_parquet(df: &DataFrame, path: &str) -> Result<()> {
    use arrow::datatypes::{Field, Schema};
    use arrow::record_batch::RecordBatch;

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
