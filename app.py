from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import pandas as pd
import numpy as np
import io, os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from werkzeug.utils import secure_filename
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.graphics.shapes import Drawing, String, Rect, Line

app = Flask(__name__)
app.secret_key = "student-marks-prediction-local-app"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

state = {"df": None, "model": None, "features": [], "target": None,
         "task": "regression", "metrics": None, "preview": None}

def read_csv(file):
    try:
        file.stream.seek(0)
        return pd.read_csv(file)
    except Exception as exc:
        raise ValueError(f"Could not read CSV: {exc}")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        uploaded = request.files.get("dataset")
        if not uploaded or not uploaded.filename:
            flash("Choose a CSV file first.", "error")
            return redirect(url_for("index"))
        if not uploaded.filename.lower().endswith(".csv"):
            flash("Please upload a .csv file.", "error")
            return redirect(url_for("index"))
        try:
            df = read_csv(uploaded)
            if df.empty or len(df.columns) < 2:
                raise ValueError("The CSV must contain data and at least two columns.")
            df = df.dropna(how="all")
            state.update(df=df, model=None, features=[], target=None, metrics=None, preview=None)
            flash(f"Loaded {len(df)} rows and {len(df.columns)} columns.", "success")
            return redirect(url_for("configure"))
        except Exception as exc:
            flash(str(exc), "error")
    return render_template("index.html")

@app.route("/configure", methods=["GET", "POST"])
def configure():
    df = state["df"]
    if df is None:
        flash("Upload a dataset first.", "error")
        return redirect(url_for("index"))
    if request.method == "POST":
        target = request.form.get("target")
        features = request.form.getlist("features")
        task = request.form.get("task", "regression")
        if not target or target not in df.columns:
            flash("Select a valid target column.", "error")
            return redirect(url_for("configure"))
        features = [c for c in features if c in df.columns and c != target]
        if not features:
            flash("Select at least one input feature column.", "error")
            return redirect(url_for("configure"))
        work = df[features + [target]].copy()
        work[target] = pd.to_numeric(work[target], errors="coerce") if task == "regression" else work[target]
        work = work.dropna(subset=[target])
        if len(work) < 4:
            flash("At least 4 usable rows are required to train the model.", "error")
            return redirect(url_for("configure"))
        X, y = work[features], work[target]
        numeric = X.select_dtypes(include=["number"]).columns.tolist()
        categorical = [c for c in features if c not in numeric]
        transformers = []
        if numeric:
            transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric))
        if categorical:
            transformers.append(("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical))
        preprocessor = ColumnTransformer(transformers)
        estimator = (RandomForestRegressor(n_estimators=180, random_state=42)
                     if task == "regression" else RandomForestClassifier(n_estimators=180, random_state=42))
        model = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        test_size = 0.2 if len(work) >= 10 else 0.25
        stratify = y if task == "classification" and y.value_counts().min() >= 2 else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=stratify)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            if task == "regression":
                metrics = {"MAE": round(float(mean_absolute_error(y_test, pred)), 3),
                           "RMSE": round(float(np.sqrt(mean_squared_error(y_test, pred))), 3)}
                if len(y_test) > 1:
                    metrics["R² score"] = round(float(r2_score(y_test, pred)), 3)
            else:
                metrics = {"Accuracy": f"{accuracy_score(y_test, pred)*100:.2f}%"}
            state.update(model=model, features=features, target=target, task=task,
                         metrics=metrics, preview=None)
            flash("Model trained successfully. You can now predict marks.", "success")
            return redirect(url_for("predict"))
        except Exception as exc:
            flash(f"Training failed: {exc}", "error")
    return render_template("configure.html", columns=df.columns.tolist(),
                           rows=len(df), preview=df.head(8).to_html(classes="table", index=False))

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if state["model"] is None:
        flash("Train a model before making predictions.", "error")
        return redirect(url_for("configure"))
    result_html = None
    chart_labels = []
    chart_values = []
    if request.method == "POST":
        mode = request.form.get("mode", "csv")
        try:
            if mode == "manual":
                row = {}
                for col in state["features"]:
                    raw = request.form.get("manual_" + col, "")
                    if raw == "":
                        row[col] = np.nan
                    else:
                        original = state["df"][col]
                        if pd.api.types.is_numeric_dtype(original):
                            row[col] = float(raw)
                        else:
                            row[col] = raw
                predict_df = pd.DataFrame([row], columns=state["features"])
            else:
                upload = request.files.get("predict_file")
                if not upload or not upload.filename:
                    raise ValueError("Choose a prediction CSV file.")
                if not upload.filename.lower().endswith(".csv"):
                    raise ValueError("Prediction file must be a CSV.")
                predict_df = read_csv(upload)
                missing = [c for c in state["features"] if c not in predict_df.columns]
                if missing:
                    raise ValueError("Prediction CSV is missing required columns: " + ", ".join(missing))
                predict_df = predict_df[state["features"]].copy()
            predictions = state["model"].predict(predict_df)
            output = predict_df.copy()
            output["Predicted " + state["target"]] = predictions
            state["preview"] = output
            result_html = output.to_html(classes="table", index=False, border=0)
            chart_labels = [f"Student {i + 1}" for i in range(len(predictions))]
            chart_values = [float(v) if isinstance(v, (int, float, np.integer, np.floating)) else str(v)
                            for v in predictions]
        except Exception as exc:
            flash(str(exc), "error")
    return render_template("predict.html", features=state["features"], target=state["target"],
                           task=state["task"], metrics=state["metrics"], result=result_html,
                           chart_labels=chart_labels, chart_values=chart_values,
                           numeric_features=[c for c in state["features"]
                               if pd.api.types.is_numeric_dtype(state["df"][c])])

@app.route("/download")
def download():
    if state["preview"] is None:
        flash("Make predictions before downloading results.", "error")
        return redirect(url_for("predict"))
    data = state["preview"].to_csv(index=False).encode("utf-8-sig")
    return send_file(io.BytesIO(data), mimetype="text/csv",
                     as_attachment=True, download_name="student_marks_predictions.csv")

@app.route("/download-pdf")
def download_pdf():
    """Download a PDF report for the latest prediction results."""
    if state["preview"] is None:
        flash("Make predictions before downloading the PDF report.", "error")
        return redirect(url_for("predict"))

    output = state["preview"].copy()
    predicted_column = "Predicted " + str(state["target"])
    page_size = landscape(A4) if len(output.columns) > 4 else A4
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=page_size,
        rightMargin=16*mm, leftMargin=16*mm,
        topMargin=16*mm, bottomMargin=16*mm
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", parent=styles["Title"],
        textColor=colors.HexColor("#4338CA"), fontSize=21, leading=26
    ))
    styles.add(ParagraphStyle(
        name="SmallNote", parent=styles["Normal"],
        textColor=colors.HexColor("#64748B"), fontSize=8
    ))

    story = [
        Paragraph("Student Marks Prediction Report", styles["ReportTitle"]),
        Paragraph("Generated on " + datetime.now().strftime("%d %B %Y, %I:%M %p"),
                  styles["SmallNote"]),
        Spacer(1, 7*mm),
        Paragraph("Prediction Summary", styles["Heading2"]),
        Paragraph(
            f"Total predicted records: <b>{len(output)}</b><br/>"
            f"Target column: <b>{state['target']}</b><br/>"
            f"Input features: <b>{', '.join(state['features'])}</b><br/>"
            f"Model type: <b>{state['task'].title()}</b>",
            styles["BodyText"]
        ),
        Spacer(1, 5*mm),
        Paragraph("Model Evaluation", styles["Heading2"])
    ]

    metric_rows = [["Metric", "Value"]]
    for name, value in (state.get("metrics") or {}).items():
        metric_rows.append([str(name), str(value)])
    if len(metric_rows) == 1:
        metric_rows.append(["Not available", ""])
    metric_table = Table(metric_rows, colWidths=[50*mm, 50*mm])
    metric_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4338CA")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F1F5F9")]),
        ("PADDING", (0, 0), (-1, -1), 7)
    ]))
    story.extend([metric_table, Spacer(1, 6*mm)])

    # A compact chart of the first 20 numeric predictions.
    if predicted_column in output.columns and pd.api.types.is_numeric_dtype(output[predicted_column]):
        values = pd.to_numeric(output[predicted_column], errors="coerce").dropna().tolist()
        if values:
            values = values[:20]
            drawing = Drawing(165*mm, 65*mm)
            left, bottom = 12*mm, 12*mm
            plot_w, plot_h = 145*mm, 45*mm
            max_value = max(max(values), 1)
            gap = 2
            bar_w = max(3, (plot_w - gap*len(values))/len(values))
            drawing.add(String(left, 62*mm, f"Predicted {state['target']} (first {len(values)} records)",
                               fontSize=9, fillColor=colors.HexColor("#334155")))
            drawing.add(Line(left, bottom, left+plot_w, bottom,
                             strokeColor=colors.HexColor("#94A3B8")))
            for i, value in enumerate(values):
                height = max(1, (float(value)/max_value)*(plot_h-4))
                x = left + i*(bar_w+gap)
                drawing.add(Rect(x, bottom, bar_w, height,
                                 fillColor=colors.HexColor("#6554D9"), strokeColor=None))
                if len(values) <= 12:
                    drawing.add(String(x+bar_w/2, bottom-8, str(i+1),
                                       textAnchor="middle", fontSize=6,
                                       fillColor=colors.HexColor("#475569")))
            story.extend([Paragraph("Prediction Chart", styles["Heading2"]),
                          drawing, Spacer(1, 5*mm)])

    story.append(Paragraph("Student-wise Prediction Results", styles["Heading2"]))
    table_data = [[str(c) for c in output.columns]]
    for row in output.itertuples(index=False, name=None):
        table_data.append(["" if pd.isna(v) else str(v) for v in row])

    usable_width = page_size[0] - 32*mm
    col_width = usable_width / max(1, len(output.columns))
    results_table = Table(table_data, repeatRows=1,
                          colWidths=[col_width] * len(output.columns))
    results_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4338CA")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F8FAFC")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 5)
    ]))
    story.append(results_table)

    doc.build(story)
    buffer.seek(0)
    return send_file(buffer, mimetype="application/pdf", as_attachment=True,
                     download_name="student_marks_prediction_report.pdf")


@app.route("/sample")
def sample():
    sample_df = pd.DataFrame({
        "Student_ID": [101,102,103,104,105,106,107,108,109,110,111,112],
        "Study_Hours": [2,3,1,5,4,6,2,7,3,8,5,6],
        "Attendance": [65,70,55,85,80,92,60,95,75,98,88,90],
        "Previous_Marks": [45,50,35,65,60,75,40,82,55,88,70,78],
        "Assignment_Score": [50,55,40,75,70,85,45,90,60,95,80,86],
        "Final_Marks": [42,48,32,68,63,79,38,85,57,92,74,81]
    })
    return send_file(io.BytesIO(sample_df.to_csv(index=False).encode("utf-8-sig")),
                     mimetype="text/csv", as_attachment=True, download_name="sample_student_marks.csv")

if __name__ == "__main__":
    app.run(debug=True)
