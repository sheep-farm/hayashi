use greeners::{Column, DataFrame};
use hayashi_lang::lang::interpreter::value::Value;
use jupyter_protocol::{Media, MediaType};

/// Convert a Hayashi `Value` into a Jupyter MIME bundle.
pub fn value_to_media(value: &Value) -> Media {
    let mut content = Vec::new();
    match value {
        Value::Plot { spec, format } => plot_to_media(spec, format, &mut content),
        Value::DataFrame(df) => dataframe_to_media(df, &mut content),
        _ => content.push(MediaType::Plain(format!("{value}"))),
    }
    Media::new(content)
}

fn plot_to_media(spec: &str, format: &str, content: &mut Vec<MediaType>) {
    match format {
        "svg" | "svg+xml" => {
            content.push(MediaType::Svg(spec.to_string()));
            content.push(MediaType::Plain("Plot(svg)".to_string()));
        }
        "png" => {
            content.push(MediaType::Png(spec.to_string()));
            content.push(MediaType::Plain("Plot(png)".to_string()));
        }
        "html" => {
            content.push(MediaType::Html(spec.to_string()));
            content.push(MediaType::Plain("Plot(html)".to_string()));
        }
        "markdown" => {
            content.push(MediaType::Markdown(spec.to_string()));
            content.push(MediaType::Plain("Plot(markdown)".to_string()));
        }
        "latex" => {
            content.push(MediaType::Latex(spec.to_string()));
            content.push(MediaType::Plain("LaTeX".to_string()));
        }
        "json" | "vega-lite" | "vega" => {
            if let Ok(obj) =
                serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(spec)
            {
                content.push(MediaType::Json(serde_json::Value::Object(obj)));
            }
            content.push(MediaType::Plain(format!("Plot({format})")));
        }
        _ => {
            content.push(MediaType::Plain(format!("Plot({format})")));
        }
    }
}

fn dataframe_to_media(df: &DataFrame, content: &mut Vec<MediaType>) {
    content.push(MediaType::Plain(format!("{df}")));
    content.push(MediaType::Html(df_to_html(df)));
}

fn df_to_html(df: &DataFrame) -> String {
    let names = df.column_names();
    if names.is_empty() {
        return "<table><tr><td>Empty DataFrame</td></tr></table>".to_string();
    }

    let n_rows = df.n_rows();
    let mut html =
        String::from("<table border=\"1\" style=\"border-collapse: collapse;\">\n<thead>\n<tr>");
    for name in &names {
        html.push_str(&format!("<th>{name}</th>"));
    }
    html.push_str("</tr>\n</thead>\n<tbody>\n");

    for i in 0..n_rows.min(100) {
        html.push_str("<tr>");
        for name in &names {
            match df.get_column(name) {
                Ok(Column::Float(arr)) => html.push_str(&format!("<td>{}</td>", fmt_f64(arr[i]))),
                Ok(Column::Int(arr)) => html.push_str(&format!("<td>{}</td>", arr[i])),
                Ok(Column::Bool(arr)) => html.push_str(&format!("<td>{}</td>", arr[i])),
                Ok(Column::String(arr)) => {
                    html.push_str(&format!("<td>{}</td>", html_escape(&arr[i])))
                }
                Ok(Column::DateTime(arr)) => html.push_str(&format!("<td>{}</td>", arr[i])),
                Ok(Column::Categorical(cat)) => html.push_str(&format!(
                    "<td>{}</td>",
                    html_escape(cat.get_string(i).unwrap_or("NA"))
                )),
                Err(_) => html.push_str("<td></td>"),
            }
        }
        html.push_str("</tr>\n");
    }

    if n_rows > 100 {
        html.push_str(&format!(
            "<tr><td colspan=\"{}\">... {n_rows} rows total ...</td></tr>\n",
            names.len()
        ));
    }

    html.push_str("</tbody>\n</table>");
    html
}

fn fmt_f64(v: f64) -> String {
    if v.is_nan() {
        ".".to_string()
    } else if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{:.0}", v)
    } else {
        format!("{:.6}", v)
    }
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn html_escape_basic() {
        assert_eq!(html_escape("a & b"), "a &amp; b");
    }
}
