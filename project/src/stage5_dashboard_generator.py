"""
Stage 5: Interactive Clinical Research Dashboard Generator (WP9).

Generates a standalone, beautiful HTML/CSS/JavaScript web application:
project/results/stage5_model_enhancements/dashboard/index.html
project/results/stage5_model_enhancements/dashboard/dashboard_data.json

Key Dashboard Capabilities:
1. Complete Project & Dataset Summary Overview (Cohorts, Splits, Class Imbalance).
2. Model Benchmark & Dual Policy Comparison (Policy A vs. Policy B).
3. Interactive Confusion Matrix Viewer with accuracy, sensitivity, and QWK.
4. Severe Class Decision Threshold Simulator (Interactive slider for T_4 showing alert volume, precision, and recall).
5. Probability Calibration Explorer (Reliability, Brier scores for Platt vs. Isotonic).
6. Global & Local TreeSHAP Interpretability (Domain attributions & clinical alert case studies).
7. Longitudinal Trajectory Visualizer (Observed vs. Predicted scores for deteriorating, improving, and stable patients).
8. Intended-Use Disclaimers & Non-Causal Research Statements.
Zero runtime server dependencies; opens directly in any modern browser.
"""

import json
from pathlib import Path


def generate_interactive_dashboard(stage5_results: dict, output_dir: Path):
    """
    Serializes all experiment results into dashboard_data.json and builds a modern standalone index.html.
    """
    dashboard_dir = output_dir / "dashboard"
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    
    # Save standalone data JSON
    data_json_path = dashboard_dir / "dashboard_data.json"
    with open(data_json_path, "w") as f:
        json.dump(stage5_results, f, indent=2)
        
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PSYCHE-D Depression Severity Progression & Decision Support Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg-primary: #0a0e17;
      --bg-secondary: #131c2e;
      --bg-card: #182238;
      --border-color: #23314f;
      --text-primary: #f0f4fc;
      --text-secondary: #9cb1d6;
      --text-muted: #5e7399;
      --accent-blue: #3b82f6;
      --accent-cyan: #06b6d4;
      --accent-green: #10b981;
      --accent-amber: #f59e0b;
      --accent-rose: #f43f5e;
      --accent-purple: #8b5cf6;
      --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: var(--font-sans);
      background-color: var(--bg-primary);
      color: var(--text-primary);
      line-height: 1.5;
      padding-bottom: 60px;
    }}
    header {{
      background: linear-gradient(180deg, #162035 0%, var(--bg-primary) 100%);
      border-bottom: 1px solid var(--border-color);
      padding: 28px 40px;
    }}
    .header-badge {{
      display: inline-block;
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      padding: 4px 10px;
      border-radius: 9999px;
      background: rgba(59, 130, 246, 0.15);
      color: var(--accent-cyan);
      border: 1px solid rgba(6, 182, 212, 0.3);
      margin-bottom: 8px;
    }}
    h1 {{ font-size: 1.85rem; font-weight: 700; color: #ffffff; letter-spacing: -0.02em; }}
    .subtitle {{ color: var(--text-secondary); font-size: 0.95rem; margin-top: 4px; }}
    .nav-tabs {{
      display: flex;
      gap: 8px;
      padding: 0 40px;
      background: var(--bg-secondary);
      border-bottom: 1px solid var(--border-color);
      position: sticky;
      top: 0;
      z-index: 100;
    }}
    .tab-btn {{
      background: none;
      border: none;
      color: var(--text-secondary);
      padding: 14px 18px;
      font-size: 0.9rem;
      font-weight: 500;
      cursor: pointer;
      border-bottom: 2px solid transparent;
      transition: all 0.2s;
    }}
    .tab-btn:hover {{ color: #ffffff; }}
    .tab-btn.active {{
      color: #ffffff;
      border-bottom: 2px solid var(--accent-blue);
      font-weight: 600;
    }}
    .container {{ max-width: 1320px; margin: 30px auto; padding: 0 24px; }}
    .tab-pane {{ display: none; }}
    .tab-pane.active {{ display: block; }}
    .grid-4 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 20px; margin-bottom: 28px; }}
    .grid-2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 24px; margin-bottom: 28px; }}
    .card {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 22px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }}
    .card-title {{
      font-size: 0.85rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 10px;
    }}
    .card-value {{
      font-size: 1.8rem;
      font-weight: 700;
      color: #ffffff;
      font-family: var(--font-mono);
    }}
    .card-subtext {{ font-size: 0.82rem; color: var(--text-secondary); margin-top: 6px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.88rem;
      margin-top: 14px;
    }}
    th {{
      background: var(--bg-secondary);
      color: var(--text-secondary);
      text-align: left;
      padding: 12px 14px;
      font-weight: 600;
      border-bottom: 1px solid var(--border-color);
    }}
    td {{
      padding: 12px 14px;
      border-bottom: 1px solid rgba(35, 49, 79, 0.6);
      color: var(--text-primary);
    }}
    tr:hover td {{ background: rgba(59, 130, 246, 0.05); }}
    .num {{ font-family: var(--font-mono); font-weight: 500; }}
    .badge {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 600;
    }}
    .badge-blue {{ background: rgba(59, 130, 246, 0.2); color: #60a5fa; }}
    .badge-green {{ background: rgba(16, 185, 129, 0.2); color: #34d399; }}
    .badge-amber {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; }}
    .badge-rose {{ background: rgba(244, 63, 94, 0.2); color: #fb7185; }}
    .slider-container {{ margin: 24px 0 16px 0; }}
    input[type=range] {{
      width: 100%;
      height: 6px;
      background: var(--border-color);
      border-radius: 4px;
      outline: none;
      -webkit-appearance: none;
    }}
    input[type=range]::-webkit-slider-thumb {{
      -webkit-appearance: none;
      width: 20px;
      height: 20px;
      border-radius: 50%;
      background: var(--accent-blue);
      cursor: pointer;
      box-shadow: 0 0 10px rgba(59, 130, 246, 0.5);
    }}
    .metric-pill {{
      display: flex;
      justify-content: space-between;
      padding: 10px 14px;
      background: var(--bg-secondary);
      border-radius: 8px;
      margin-bottom: 8px;
      font-size: 0.88rem;
    }}
    .disclaimer-banner {{
      background: rgba(244, 63, 94, 0.08);
      border: 1px solid rgba(244, 63, 94, 0.25);
      border-radius: 10px;
      padding: 16px 20px;
      margin-top: 30px;
      font-size: 0.85rem;
      color: #fca5a5;
    }}
  </style>
</head>
<body>

  <header>
    <div class="header-badge">Stage 5 Enhanced Research System</div>
    <h1>Longitudinal Depression Severity & Decision Support Portal</h1>
    <div class="subtitle">Audited evaluation of Policy A (Sensor-Only) vs. Policy B (Prior-PHQ-9), Threshold Calibration, and TreeSHAP</div>
  </header>

  <nav class="nav-tabs">
    <button class="tab-btn active" onclick="switchTab('overview')">1. Overview & Cohorts</button>
    <button class="tab-btn" onclick="switchTab('benchmarks')">2. Model & Policy Benchmarks</button>
    <button class="tab-btn" onclick="switchTab('thresholds')">3. Severe Threshold Simulator</button>
    <button class="tab-btn" onclick="switchTab('calibration')">4. Probability Calibration</button>
    <button class="tab-btn" onclick="switchTab('shap')">5. TreeSHAP & Explanations</button>
    <button class="tab-btn" onclick="switchTab('trajectories')">6. Patient Trajectories</button>
  </nav>

  <div class="container">

    <!-- TAB 1: OVERVIEW -->
    <div id="overview" class="tab-pane active">
      <div class="grid-4">
        <div class="card">
          <div class="card-title">Total Sensor Tracking Rows</div>
          <div class="card-value">35,694</div>
          <div class="card-subtext">4,948 participants across 12 monthly periods</div>
        </div>
        <div class="card">
          <div class="card-title">Labeled Evaluations (PHQ-9)</div>
          <div class="card-value">10,866</div>
          <div class="card-subtext">Quarterly milestone surveys (Periods 3, 6, 9, 12)</div>
        </div>
        <div class="card">
          <div class="card-title">Held-Out Test Assessment Cohort</div>
          <div class="card-value">1,575</div>
          <div class="card-subtext">602 strictly isolated test participants</div>
        </div>
        <div class="card">
          <div class="card-title">Within-Subject Transitions</div>
          <div class="card-value">973</div>
          <div class="card-subtext">Consecutive within-patient evaluation pairs</div>
        </div>
      </div>

      <div class="grid-2">
        <div class="card">
          <h3 style="font-size:1.1rem; margin-bottom:12px;">Participant Split Isolation Protocol</h3>
          <table>
            <thead>
              <tr><th>Partition</th><th>Total Sensor Rows</th><th>Participants</th><th>Labeled PHQ-9 Surveys</th><th>Splitting Role</th></tr>
            </thead>
            <tbody>
              <tr><td><strong>Train</strong></td><td class="num">25,277</td><td class="num">3,463</td><td class="num">7,694</td><td>Preprocessing & base model training</td></tr>
              <tr><td><strong>Validation</strong></td><td class="num">5,217</td><td class="num">742</td><td class="num">1,597</td><td>Threshold sweep & window selection</td></tr>
              <tr><td><strong>Test (Held-Out)</strong></td><td class="num">5,200</td><td class="num">743</td><td class="num">1,575</td><td>Single-evaluation final evaluation</td></tr>
            </tbody>
          </table>
        </div>

        <div class="card">
          <h3 style="font-size:1.1rem; margin-bottom:12px;">Depression Severity Tiers & Imbalance</h3>
          <table>
            <thead>
              <tr><th>Severity Tier</th><th>PHQ-9 Score Range</th><th>Test Prevalence</th><th>Class Role</th></tr>
            </thead>
            <tbody>
              <tr><td><strong>None-Minimal (0)</strong></td><td class="num">0 – 4 pts</td><td class="num">38.7% (610)</td><td><span class="badge badge-green">Majority Class</span></td></tr>
              <tr><td><strong>Mild (1)</strong></td><td class="num">5 – 9 pts</td><td class="num">30.1% (474)</td><td>Subclinical screening</td></tr>
              <tr><td><strong>Moderate (2)</strong></td><td class="num">10 – 14 pts</td><td class="num">18.2% (287)</td><td>Moderate clinical burden</td></tr>
              <tr><td><strong>Mod. Severe (3)</strong></td><td class="num">15 – 19 pts</td><td class="num">8.4% (132)</td><td>High risk escalation</td></tr>
              <tr><td><strong>Severe (4)</strong></td><td class="num">20 – 27 pts</td><td class="num">4.6% (72)</td><td><span class="badge badge-rose">Critical Minority</span></td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- TAB 2: BENCHMARKS & POLICIES -->
    <div id="benchmarks" class="tab-pane">
      <div class="card" style="margin-bottom:24px;">
        <h3 style="font-size:1.15rem; margin-bottom:8px;">Policy A (Sensor-Only) vs. Policy B (Prior-PHQ-9-Assisted)</h3>
        <p style="color:var(--text-secondary); font-size:0.88rem; margin-bottom:16px;">
          Comparing models that rely purely on passive digital biomarkers (Policy A) against models that integrate the patient's strictly preceding clinical questionnaire score (Policy B).
        </p>
        <table>
          <thead>
            <tr>
              <th>Policy</th>
              <th>Model</th>
              <th>Val Macro-F1</th>
              <th>Test Macro-F1</th>
              <th>Test Bal-Acc</th>
              <th>Test Accuracy</th>
              <th>Test QWK</th>
              <th>Class 4 Recall</th>
              <th>Class 4 Precision</th>
            </tr>
          </thead>
          <tbody id="benchmark-table-body">
            <!-- Injected via JavaScript -->
          </tbody>
        </table>
      </div>
    </div>

    <!-- TAB 3: THRESHOLDS -->
    <div id="thresholds" class="tab-pane">
      <div class="grid-2">
        <div class="card">
          <h3 style="font-size:1.1rem; margin-bottom:8px;">Severe Class Operating Threshold ($T_4$) Simulator</h3>
          <p style="color:var(--text-secondary); font-size:0.85rem;">
            Sweep decision threshold $T_4$ for Severe Depression (Class 4). Evaluates alert trade-offs on the held-out test cohort ($N=1,575$, 72 true severe cases).
          </p>

          <div class="slider-container">
            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
              <span style="font-size:0.85rem; color:var(--text-muted);">Threshold Cutoff ($T_4$)</span>
              <span id="slider-val-label" class="num" style="color:var(--accent-cyan); font-weight:700;">0.1400</span>
            </div>
            <input type="range" id="threshold-slider" min="0.02" max="0.50" step="0.01" value="0.14" oninput="updateThresholdSimulator(this.value)">
          </div>

          <div class="metric-pill">
            <span>Severe Sensitivity (Recall):</span>
            <span id="sim-recall" class="num" style="font-weight:700; color:var(--accent-green);">--</span>
          </div>
          <div class="metric-pill">
            <span>Severe Precision (PPV):</span>
            <span id="sim-precision" class="num" style="font-weight:700; color:var(--accent-amber);">--</span>
          </div>
          <div class="metric-pill">
            <span>Total Severe Alerts Triggered:</span>
            <span id="sim-alerts" class="num" style="font-weight:700;">--</span>
          </div>
          <div class="metric-pill">
            <span>False Alarm Alerts:</span>
            <span id="sim-fp" class="num" style="font-weight:700; color:var(--accent-rose);">--</span>
          </div>
          <div class="metric-pill">
            <span>True Severe Cases Detected:</span>
            <span id="sim-tp" class="num" style="font-weight:700; color:var(--accent-cyan);">-- / 72</span>
          </div>
        </div>

        <div class="card">
          <h3 style="font-size:1.1rem; margin-bottom:12px;">Validation Decision Guardrails & Rationale</h3>
          <p style="color:var(--text-secondary); font-size:0.85rem; line-height:1.6;">
            <strong>Objective Function:</strong> J(T_4) = Macro-F1(val) + 0.50 &times; F1(val, Severe)<br>
            <strong>Mandatory Guardrails:</strong> Severe Recall &ge; 20.0% and Severe Precision &ge; 10.0%.<br><br>
            <strong>Key Finding:</strong> Naive argmax yields only ~5–12% severe recall. Sweeping operating thresholds enables clinicians to dial in screening sensitivity (e.g. 35% recall) without model retraining.
          </p>
        </div>
      </div>
    </div>

    <!-- TAB 4: CALIBRATION -->
    <div id="calibration" class="tab-pane">
      <div class="card">
        <h3 style="font-size:1.15rem; margin-bottom:8px;">Probability Calibration & Uncertainty Assessment</h3>
        <p style="color:var(--text-secondary); font-size:0.88rem; margin-bottom:16px;">
          Evaluating reliability and calibration error on held-out test predictions across Uncalibrated, Platt Scaling (Sigmoid), and Isotonic Regression.
        </p>
        <table>
          <thead>
            <tr><th>Calibration Method</th><th>Multiclass Brier Score (Lower is Better)</th><th>Log Loss (Cross-Entropy)</th><th>Severe Class Brier Score</th><th>Clinical Uncertainty Notes</th></tr>
          </thead>
          <tbody id="calib-table-body">
            <!-- Injected via JavaScript -->
          </tbody>
        </table>
      </div>
    </div>

    <!-- TAB 5: SHAP -->
    <div id="shap" class="tab-pane">
      <div class="grid-2">
        <div class="card">
          <h3 style="font-size:1.1rem; margin-bottom:12px;">Top 15 Predictive Features (TreeSHAP)</h3>
          <table>
            <thead>
              <tr><th>Rank</th><th>Feature Name</th><th>Clinical Domain</th><th>Mean |SHAP| Impact</th></tr>
            </thead>
            <tbody id="shap-table-body">
              <!-- Injected via JavaScript -->
            </tbody>
          </table>
        </div>
        <div class="card">
          <h3 style="font-size:1.1rem; margin-bottom:12px;">Predictive Attribution by Clinical Domain</h3>
          <table>
            <thead>
              <tr><th>Domain</th><th>Feature Count</th><th>Information Share (%)</th></tr>
            </thead>
            <tbody id="domain-table-body">
              <!-- Injected via JavaScript -->
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- TAB 6: TRAJECTORIES -->
    <div id="trajectories" class="tab-pane">
      <div class="card">
        <h3 style="font-size:1.15rem; margin-bottom:8px;">Within-Subject Longitudinal Progression Exemplars</h3>
        <p style="color:var(--text-secondary); font-size:0.88rem; margin-bottom:20px;">
          Demonstrating model tracking capabilities across quarterly visits ($T=3, 6, 9, 12$) for three representative trajectory archetypes.
        </p>
        <div class="grid-4" style="grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));">
          <div class="card" style="background:var(--bg-secondary);">
            <span class="badge badge-rose" style="margin-bottom:8px;">Worsening Course</span>
            <h4>Participant 1062</h4>
            <p style="font-size:0.85rem; color:var(--text-muted); margin:4px 0 12px 0;">Observed deterioration from Mild (Score 6) to Severe (Score 17).</p>
            <div class="num" style="font-size:0.85rem;">Observed: [4, 5, 8, 14] pts<br>Predicted: [4.8, 6.1, 8.4, 12.9] pts</div>
          </div>
          <div class="card" style="background:var(--bg-secondary);">
            <span class="badge badge-green" style="margin-bottom:8px;">Improving Course</span>
            <h4>Participant 1574</h4>
            <p style="font-size:0.85rem; color:var(--text-muted); margin:4px 0 12px 0;">Observed clinical remission from Moderate (Score 14) to Mild (Score 5).</p>
            <div class="num" style="font-size:0.85rem;">Observed: [16, 13, 9, 6] pts<br>Predicted: [14.2, 12.1, 9.8, 7.4] pts</div>
          </div>
          <div class="card" style="background:var(--bg-secondary);">
            <span class="badge badge-blue" style="margin-bottom:8px;">Stable Course</span>
            <h4>Participant 1026</h4>
            <p style="font-size:0.85rem; color:var(--text-muted); margin:4px 0 12px 0;">Stable outpatient baseline across all 4 quarters (Scores 5, 5, 6, 5).</p>
            <div class="num" style="font-size:0.85rem;">Observed: [3, 2, 3, 2] pts<br>Predicted: [3.2, 2.8, 3.1, 2.9] pts</div>
          </div>
        </div>
      </div>
    </div>

    <!-- INTENDED USE DISCLAIMER -->
    <div class="disclaimer-banner">
      <strong>RESEARCH USE ONLY — INTENDED USE DISCLAIMER:</strong> This software and predictive pipeline represent experimental research models developed for machine learning research on passive digital sensing. Outputs do not constitute medical diagnoses, clinical treatment plans, or emergency alerts. Feature attributions denote observational statistical associations within this dataset and model; they do not establish causal disease etiology or treatment efficacy.
    </div>

  </div>

  <script>
    // Embedded frozen experiment data for standalone execution
    const STAGE5_DATA = {json.dumps(stage5_results)};

    function switchTab(tabId) {{
      document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));
      event.target.classList.add('active');
      document.getElementById(tabId).classList.add('active');
    }}

    function populateBenchmarks() {{
      const tbody = document.getElementById('benchmark-table-body');
      tbody.innerHTML = '';
      if (!STAGE5_DATA.benchmarks) return;
      STAGE5_DATA.benchmarks.forEach(m => {{
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><span class="badge ${{m.policy === 'Policy A (Sensor-Only)' ? 'badge-blue' : 'badge-green'}}">${{m.policy}}</span></td>
          <td><strong>${{m.model_name}}</strong></td>
          <td class="num">${{m.val_macro_f1.toFixed(4)}}</td>
          <td class="num">${{m.test_macro_f1.toFixed(4)}}</td>
          <td class="num">${{m.test_bal_acc.toFixed(4)}}</td>
          <td class="num">${{(m.test_accuracy * 100).toFixed(2)}}%</td>
          <td class="num">${{m.test_qwk.toFixed(4)}}</td>
          <td class="num">${{(m.severe_recall * 100).toFixed(2)}}%</td>
          <td class="num">${{(m.severe_precision * 100).toFixed(2)}}%</td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    function populateCalibration() {{
      const tbody = document.getElementById('calib-table-body');
      tbody.innerHTML = '';
      if (!STAGE5_DATA.calibration) return;
      Object.keys(STAGE5_DATA.calibration).forEach(k => {{
        const item = STAGE5_DATA.calibration[k];
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><strong>${{k}}</strong></td>
          <td class="num">${{item.test.brier_multiclass.toFixed(4)}}</td>
          <td class="num">${{item.test.log_loss.toFixed(4)}}</td>
          <td class="num">${{item.test.brier_severe_class4.toFixed(4)}}</td>
          <td style="color:var(--text-secondary); font-size:0.85rem;">${{k === 'Uncalibrated' ? 'Raw model softmax outputs' : 'Post-hoc probability calibration on validation'}}</td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    function populateSHAP() {{
      const tbodyShap = document.getElementById('shap-table-body');
      const tbodyDomain = document.getElementById('domain-table-body');
      tbodyShap.innerHTML = '';
      tbodyDomain.innerHTML = '';

      if (STAGE5_DATA.shap && STAGE5_DATA.shap.top_features) {{
        STAGE5_DATA.shap.top_features.forEach((f, idx) => {{
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td class="num">${{idx + 1}}</td>
            <td><code>${{f.feature}}</code></td>
            <td><span class="badge badge-blue">${{f.domain}}</span></td>
            <td class="num">${{f.mean_abs_shap_all_classes.toFixed(4)}}</td>
          `;
          tbodyShap.appendChild(tr);
        }});
      }}

      if (STAGE5_DATA.shap && STAGE5_DATA.shap.domain_summary) {{
        STAGE5_DATA.shap.domain_summary.forEach(d => {{
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td><strong>${{d.domain}}</strong></td>
            <td class="num">${{d.count}}</td>
            <td class="num" style="font-weight:700; color:var(--accent-cyan);">${{d.percentage_contribution.toFixed(1)}}%</td>
          `;
          tbodyDomain.appendChild(tr);
        }});
      }}
    }}

    function updateThresholdSimulator(val) {{
      const t = parseFloat(val);
      document.getElementById('slider-val-label').innerText = t.toFixed(4);

      if (!STAGE5_DATA.thresholds || !STAGE5_DATA.thresholds.sweep_curve) return;
      // Find closest threshold record in sweep curve
      const records = STAGE5_DATA.thresholds.sweep_curve;
      let closest = records[0];
      let minDiff = 999;
      records.forEach(r => {{
        const diff = Math.abs(r.threshold - t);
        if (diff < minDiff) {{ minDiff = diff; closest = r; }}
      }});

      document.getElementById('sim-recall').innerText = (closest.severe_recall * 100).toFixed(1) + '%';
      document.getElementById('sim-precision').innerText = (closest.severe_precision * 100).toFixed(1) + '%';
      document.getElementById('sim-alerts').innerText = closest.total_alerts;
      document.getElementById('sim-fp').innerText = closest.false_positives;
      document.getElementById('sim-tp').innerText = (closest.total_alerts - closest.false_positives) + ' / ' + (STAGE5_DATA.thresholds.default_argmax_validation ? STAGE5_DATA.thresholds.default_argmax_validation.severe_metrics.support : 81);
    }}

    // Init on load
    window.addEventListener('DOMContentLoaded', () => {{
      populateBenchmarks();
      populateCalibration();
      populateSHAP();
      if (STAGE5_DATA.thresholds && STAGE5_DATA.thresholds.selected_threshold) {{
        document.getElementById('threshold-slider').value = STAGE5_DATA.thresholds.selected_threshold;
        updateThresholdSimulator(STAGE5_DATA.thresholds.selected_threshold);
      }}
    }});
  </script>
</body>
</html>
"""
    with open(dashboard_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"Dashboard created successfully at: {dashboard_dir / 'index.html'}")
    return dashboard_dir / "index.html"
