"""
Stage 5: Interactive Clinical Research Dashboard Generator (WP9).

Generates a standalone, beautiful HTML/CSS/JavaScript web application:
- project/results/stage5_model_enhancements/dashboard/index.html
- project/results/stage5_model_enhancements/clinical_decision_dashboard.html
- project/results/stage5_model_enhancements/dashboard/dashboard_data.json

Key Dashboard Capabilities:
1. Complete Project & Dataset Summary Overview (Cohorts, Splits, Class Imbalance).
2. Comprehensive Longitudinal Dynamics & Progression Explorer (Interactive Causal Window Visualizer,
   Recurrent Fusion GRU Dataflow, MCID Clinical Progression Classifier, Persistence Paradox Simulator,
   and Interactive Patient Timeline Journey Simulator).
3. Model Benchmark & Dual Policy Comparison (Policy A vs. Policy B).
4. Severe Class Decision Threshold Simulator (Interactive slider for T_4 showing alert volume, precision, and recall).
5. Probability Calibration Explorer (Reliability, Brier scores for Platt vs. Isotonic).
6. Global & Local TreeSHAP Interpretability (Domain attributions & clinical alert case studies).
7. Intended-Use Disclaimers & Non-Causal Research Statements.
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
  <title>Longitudinal Depression Severity Progression & Decision Support Dashboard</title>
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
      overflow-x: auto;
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
      white-space: nowrap;
    }}
    .tab-btn:hover {{ color: #ffffff; }}
    .tab-btn.active {{
      color: #ffffff;
      border-bottom: 2px solid var(--accent-blue);
      font-weight: 600;
    }}
    .tab-btn.highlight {{
      color: var(--accent-cyan);
    }}
    .container {{ max-width: 1340px; margin: 30px auto; padding: 0 24px; }}
    .tab-pane {{ display: none; }}
    .tab-pane.active {{ display: block; }}
    .grid-4 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 20px; margin-bottom: 28px; }}
    .grid-3 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 22px; margin-bottom: 28px; }}
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
    .badge-blue {{ background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }}
    .badge-green {{ background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }}
    .badge-amber {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }}
    .badge-rose {{ background: rgba(244, 63, 94, 0.2); color: #fb7185; border: 1px solid rgba(244, 63, 94, 0.3); }}
    .badge-purple {{ background: rgba(139, 92, 246, 0.2); color: #c084fc; border: 1px solid rgba(139, 92, 246, 0.3); }}
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

    /* Longitudinal Architecture Visualizer Styles */
    .flow-step-box {{
      background: var(--bg-secondary);
      border: 1px solid var(--border-color);
      border-radius: 10px;
      padding: 16px;
      position: relative;
    }}
    .flow-step-header {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 10px;
    }}
    .step-number {{
      width: 26px;
      height: 26px;
      border-radius: 50%;
      background: var(--accent-blue);
      color: #fff;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.8rem;
      font-weight: 700;
    }}
    .timeline-scrubber {{
      display: flex;
      gap: 10px;
      margin: 16px 0;
      background: var(--bg-secondary);
      padding: 10px;
      border-radius: 10px;
      border: 1px solid var(--border-color);
    }}
    .scrub-btn {{
      flex: 1;
      padding: 12px 14px;
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      color: var(--text-secondary);
      cursor: pointer;
      text-align: center;
      transition: all 0.2s;
    }}
    .scrub-btn:hover {{ border-color: var(--accent-cyan); color: #fff; }}
    .scrub-btn.active {{
      background: rgba(59, 130, 246, 0.18);
      border-color: var(--accent-blue);
      color: #fff;
      font-weight: 600;
      box-shadow: 0 0 12px rgba(59, 130, 246, 0.3);
    }}
    .period-cell {{
      display: inline-block;
      width: 44px;
      height: 44px;
      line-height: 44px;
      text-align: center;
      border-radius: 8px;
      margin-right: 6px;
      font-size: 0.85rem;
      font-weight: 600;
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      color: var(--text-secondary);
      transition: all 0.2s;
    }}
    .period-cell.in-window {{
      background: rgba(59, 130, 246, 0.25);
      border-color: var(--accent-blue);
      color: #ffffff;
      box-shadow: 0 0 8px rgba(59, 130, 246, 0.4);
    }}
    .period-cell.target {{
      background: rgba(16, 185, 129, 0.25);
      border-color: var(--accent-green);
      color: #ffffff;
      box-shadow: 0 0 8px rgba(16, 185, 129, 0.4);
    }}
    .period-cell.quarantined {{
      background: rgba(244, 63, 94, 0.15);
      border-color: rgba(244, 63, 94, 0.4);
      color: var(--accent-rose);
      opacity: 0.6;
    }}
    .patient-pill {{
      padding: 10px 16px;
      border-radius: 8px;
      background: var(--bg-secondary);
      border: 1px solid var(--border-color);
      color: var(--text-secondary);
      cursor: pointer;
      font-size: 0.88rem;
      font-weight: 500;
      transition: all 0.2s;
    }}
    .patient-pill:hover {{ color: #fff; border-color: var(--accent-blue); }}
    .patient-pill.active {{
      background: rgba(59, 130, 246, 0.2);
      border-color: var(--accent-blue);
      color: #fff;
      font-weight: 600;
    }}
    .gauge-bar-bg {{
      background: var(--bg-primary);
      height: 8px;
      border-radius: 4px;
      overflow: hidden;
      margin-top: 6px;
    }}
    .gauge-bar-fill {{
      height: 100%;
      border-radius: 4px;
      transition: width 0.4s ease;
    }}
  </style>
</head>
<body>

  <header>
    <div class="header-badge">Stage 5 Enhanced Research System</div>
    <h1>Longitudinal Depression Severity & Decision Support Portal</h1>
    <div class="subtitle">Passive Wearable Actigraphy, Recurrent Fusion GRU, Calibrated XGBoost & Clinical Decision Support</div>
  </header>

  <nav class="nav-tabs">
    <button class="tab-btn active" onclick="switchTab('overview')">1. Cohort & Study Design</button>
    <button class="tab-btn highlight" onclick="switchTab('longitudinal')">2. 🧠 Longitudinal Dynamics Explorer</button>
    <button class="tab-btn" onclick="switchTab('benchmarks')">3. Model Benchmarks (Policy A vs. B)</button>
    <button class="tab-btn" onclick="switchTab('thresholds')">4. Severe Alert Threshold Simulator</button>
    <button class="tab-btn" onclick="switchTab('calibration')">5. Probability Calibration</button>
    <button class="tab-btn" onclick="switchTab('shap')">6. TreeSHAP & Biomarkers</button>
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
          <p style="font-size:0.82rem; color:var(--text-secondary); margin-top:12px;">
            <span class="badge badge-green">Zero Leakage</span> Enforced at subject level. Zero participant data exists in multiple splits.
          </p>
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

    <!-- TAB 2: LONGITUDINAL DYNAMICS & PROGRESSION (NEW FLAGSHIP TAB) -->
    <div id="longitudinal" class="tab-pane">
      
      <!-- HERO EXPLANATION -->
      <div class="card" style="margin-bottom:28px; background:linear-gradient(135deg, rgba(24,34,56,0.95) 0%, rgba(19,28,46,0.95) 100%); border-color:rgba(59,130,246,0.3);">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
          <div>
            <span class="badge badge-purple" style="margin-bottom:8px;">Core Architecture Breakdown</span>
            <h2 style="font-size:1.45rem; color:#fff; margin-bottom:8px;">How Longitudinal Analysis Works In This Model</h2>
            <p style="color:var(--text-secondary); font-size:0.92rem; max-width:850px; line-height:1.6;">
              Traditional depression diagnostics occur intermittently via quarterly PHQ-9 questionnaires, leaving vast "blind spots" where relapses occur undetected. 
              Our framework uses <strong>passive continuous wearable sensors</strong> (actigraphy, sleep fragmentation, circadian shifts) combined with a <strong>Recurrent Fusion GRU</strong> to detect subtle behavioral deterioration before severe clinical escalation.
            </p>
          </div>
          <div style="text-align:right;">
            <div class="num" style="font-size:1.6rem; color:var(--accent-cyan); font-weight:700;">973 Transitions</div>
            <div style="font-size:0.8rem; color:var(--text-muted);">Evaluated in Test Cohort</div>
          </div>
        </div>

        <!-- 4-STEP PIPELINE CARDS -->
        <div class="grid-4" style="margin-top:24px; margin-bottom:0;">
          <div class="flow-step-box">
            <div class="flow-step-header">
              <div class="step-number">1</div>
              <strong style="font-size:0.88rem;">Continuous Passive Sensing</strong>
            </div>
            <p style="font-size:0.82rem; color:var(--text-secondary);">
              Wearables record 150 features (steps, sedentary bouts, sleep efficiency, mid-sleep circadian times) 24/7 without patient burden.
            </p>
          </div>

          <div class="flow-step-box">
            <div class="flow-step-header">
              <div class="step-number">2</div>
              <strong style="font-size:0.88rem;">Causal Slicing ($W$)</strong>
            </div>
            <p style="font-size:0.82rem; color:var(--text-secondary);">
              Extracts preceding history windows (T-W ... T-1). Future periods (&gt; T) are strictly quarantined to guarantee zero data leakage.
            </p>
          </div>

          <div class="flow-step-box">
            <div class="flow-step-header">
              <div class="step-number">3</div>
              <strong style="font-size:0.88rem;">Recurrent Fusion GRU</strong>
            </div>
            <p style="font-size:0.82rem; color:var(--text-secondary);">
              A GRU encodes historical trajectory into h<sub>final</sub> (64-dim hidden vector) and merges with concurrent features x<sub>T</sub> via a dense fusion head.
            </p>
          </div>

          <div class="flow-step-box">
            <div class="flow-step-header">
              <div class="step-number">4</div>
              <strong style="font-size:0.88rem;">Trajectory & MCID Output</strong>
            </div>
            <p style="font-size:0.82rem; color:var(--text-secondary);">
              Outputs discrete severity (y&#770;<sub>T</sub>), continuous score (s&#770;<sub>T</sub>), within-subject change (&Delta;s), and MCID progression category.
            </p>
          </div>
        </div>
      </div>

      <!-- INTERACTIVE SECTION 1: OBSERVATION WINDOW SLICER -->
      <div class="grid-2">
        <div class="card">
          <h3 style="font-size:1.1rem; margin-bottom:8px;">1. Interactive Causal Observation Window ($W$) Slicer</h3>
          <p style="color:var(--text-secondary); font-size:0.85rem; margin-bottom:16px;">
            Choose an observation window to see how historical periods are extracted for target milestone Period $T = 6$ (Quarter 2).
          </p>

          <div style="display:flex; gap:8px; margin-bottom:18px;">
            <button class="patient-pill active" onclick="setWindowDemo(1, this)">Window W = 1 (Selected)</button>
            <button class="patient-pill" onclick="setWindowDemo(2, this)">Window W = 2</button>
            <button class="patient-pill" onclick="setWindowDemo(3, this)">Window W = 3</button>
            <button class="patient-pill" onclick="setWindowDemo('full', this)">Full History</button>
          </div>

          <div style="margin-bottom:14px;">
            <span style="font-size:0.8rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.05em;">Timeline Slicing (Period 1 to 12):</span>
            <div id="window-cells-container" style="margin-top:8px;">
              <!-- Injected via JavaScript -->
            </div>
          </div>

          <div style="display:flex; gap:16px; font-size:0.8rem; margin-top:14px;">
            <div><span class="period-cell in-window" style="width:20px; height:20px; line-height:20px;"></span> Preceding History ($H_T$)</div>
            <div><span class="period-cell target" style="width:20px; height:20px; line-height:20px;"></span> Current Period ($x_T$)</div>
            <div><span class="period-cell quarantined" style="width:20px; height:20px; line-height:20px;"></span> Quarantined Future (No Leakage)</div>
          </div>

          <div id="window-expl-text" style="background:var(--bg-secondary); border-radius:8px; padding:12px; margin-top:16px; font-size:0.82rem; color:var(--text-secondary);">
            <strong>W = 1 Configuration:</strong> Uses immediate preceding Period 5 sensor data (H<sub>T</sub> of shape 1 &times; 150) plus Period 6 current sensors to predict PHQ-9 at Month 6. Prospectively selected configuration on validation Macro-F1 (0.3359).
          </div>
        </div>

        <!-- INTERACTIVE SECTION 2: THE PERSISTENCE BASELINE PARADOX -->
        <div class="card">
          <h3 style="font-size:1.1rem; margin-bottom:8px;">2. The "Persistence Baseline Paradox" Exposed</h3>
          <p style="color:var(--text-secondary); font-size:0.85rem; margin-bottom:14px;">
            62.4% of patients remain in the same severity tier between quarters. A naive model predicting "no change" (y&#770;<sub>t</sub> = y<sub>t-1</sub>) appears deceptively accurate, but is <strong>100% blind to progression</strong>.
          </p>

          <div class="grid-2" style="margin-bottom:12px; gap:14px;">
            <div style="background:rgba(244,63,94,0.06); border:1px solid rgba(244,63,94,0.3); border-radius:10px; padding:14px;">
              <span class="badge badge-rose" style="margin-bottom:6px;">Naive Persistence Baseline</span>
              <div style="font-size:0.8rem; color:var(--text-muted); margin-bottom:10px;">Predicts patient never changes (y&#770;<sub>t</sub> = y<sub>t-1</sub>)</div>
              
              <div class="metric-pill" style="margin-bottom:6px;">
                <span>Static Macro-F1:</span>
                <span class="num" style="font-weight:700; color:#fff;">0.4285 (Deceptive!)</span>
              </div>
              <div class="metric-pill" style="margin-bottom:6px;">
                <span>Directional Agreement:</span>
                <span class="num" style="font-weight:700; color:var(--accent-rose);">0.00% (Never flags)</span>
              </div>
              <div class="metric-pill" style="margin-bottom:0;">
                <span>MCID Progression F1:</span>
                <span class="num" style="font-weight:700; color:var(--text-secondary);">0.2889</span>
              </div>
              <div style="font-size:0.78rem; color:#fca5a5; margin-top:10px;">
                ⚠️ <strong>Clinical Hazard:</strong> Misses 100% of deteriorating patients who need intervention.
              </div>
            </div>

            <div style="background:rgba(16,185,129,0.06); border:1px solid rgba(16,185,129,0.3); border-radius:10px; padding:14px;">
              <span class="badge badge-green" style="margin-bottom:6px;">Recurrent Fusion GRU Model</span>
              <div style="font-size:0.8rem; color:var(--text-muted); margin-bottom:10px;">Models dynamic sensor trajectory shifts</div>
              
              <div class="metric-pill" style="margin-bottom:6px;">
                <span>Static Macro-F1:</span>
                <span class="num" style="font-weight:700; color:#fff;">0.3546</span>
              </div>
              <div class="metric-pill" style="margin-bottom:6px;">
                <span>Directional Agreement:</span>
                <span class="num" style="font-weight:700; color:var(--accent-green);">44.60% (Active Tracking)</span>
              </div>
              <div class="metric-pill" style="margin-bottom:0;">
                <span>MCID Progression F1:</span>
                <span class="num" style="font-weight:700; color:var(--accent-cyan);">0.3367 (+16.5% gain)</span>
              </div>
              <div style="font-size:0.78rem; color:#6ee7b7; margin-top:10px;">
                ✅ <strong>Clinical Advantage:</strong> Successfully identifies patients transitioning into higher risk tiers.
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- INTERACTIVE SECTION 3: PATIENT TRAJECTORY SIMULATOR -->
      <div class="card" style="margin-top:24px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
          <div>
            <h3 style="font-size:1.15rem;">3. Interactive Patient Trajectory Simulator</h3>
            <p style="color:var(--text-secondary); font-size:0.85rem;">
              Step through consecutive quarterly visits ($Q_1 \to Q_4$) to see how continuous wearable biomarkers change, how the GRU updates its trajectory, and when clinical alerts trigger.
            </p>
          </div>
          <div style="display:flex; gap:8px;">
            <button class="patient-pill active" onclick="selectPatientTrajectory('relapse', this)">Patient #1062: Deterioration / Relapse</button>
            <button class="patient-pill" onclick="selectPatientTrajectory('remission', this)">Patient #1574: Treatment Remission</button>
            <button class="patient-pill" onclick="selectPatientTrajectory('stable', this)">Patient #1026: Outpatient Stable</button>
          </div>
        </div>

        <!-- TIMELINE SCRUBBER -->
        <div class="timeline-scrubber">
          <button class="scrub-btn active" onclick="scrubQuarter(0, this)">
            <div style="font-size:0.75rem; text-transform:uppercase;">Quarter 1 (Month 3)</div>
            <div id="scrub-q1-score" class="num" style="font-size:1.1rem; font-weight:700; margin-top:4px;">PHQ-9: 4 pts</div>
          </button>
          <button class="scrub-btn" onclick="scrubQuarter(1, this)">
            <div style="font-size:0.75rem; text-transform:uppercase;">Quarter 2 (Month 6)</div>
            <div id="scrub-q2-score" class="num" style="font-size:1.1rem; font-weight:700; margin-top:4px;">PHQ-9: 5 pts</div>
          </button>
          <button class="scrub-btn" onclick="scrubQuarter(2, this)">
            <div style="font-size:0.75rem; text-transform:uppercase;">Quarter 3 (Month 9)</div>
            <div id="scrub-q3-score" class="num" style="font-size:1.1rem; font-weight:700; margin-top:4px;">PHQ-9: 8 pts</div>
          </button>
          <button class="scrub-btn" onclick="scrubQuarter(3, this)">
            <div style="font-size:0.75rem; text-transform:uppercase;">Quarter 4 (Month 12)</div>
            <div id="scrub-q4-score" class="num" style="font-size:1.1rem; font-weight:700; margin-top:4px;">PHQ-9: 14 pts</div>
          </button>
        </div>

        <!-- LIVE METRICS & SENSORS FOR SELECTED QUARTER -->
        <div class="grid-3" style="margin-top:20px; margin-bottom:0;">
          
          <!-- SENSOR GAUGES -->
          <div class="card" style="background:var(--bg-secondary);">
            <div class="card-title">Continuous Passive Wearable Sensors</div>
            
            <div style="margin-bottom:12px;">
              <div style="display:flex; justify-content:space-between; font-size:0.82rem;">
                <span>Weekday Nocturnal Sleep:</span>
                <span id="gauge-sleep" class="num" style="font-weight:700; color:var(--accent-cyan);">7.4 hrs</span>
              </div>
              <div class="gauge-bar-bg"><div id="bar-sleep" class="gauge-bar-fill" style="width:74%; background:var(--accent-cyan);"></div></div>
            </div>

            <div style="margin-bottom:12px;">
              <div style="display:flex; justify-content:space-between; font-size:0.82rem;">
                <span>Daily Daytime Sedentary Hours:</span>
                <span id="gauge-sedentary" class="num" style="font-weight:700; color:var(--accent-amber);">7.8 hrs</span>
              </div>
              <div class="gauge-bar-bg"><div id="bar-sedentary" class="gauge-bar-fill" style="width:52%; background:var(--accent-amber);"></div></div>
            </div>

            <div style="margin-bottom:12px;">
              <div style="display:flex; justify-content:space-between; font-size:0.82rem;">
                <span>Awake Hourly Movement (Steps):</span>
                <span id="gauge-steps" class="num" style="font-weight:700; color:var(--accent-blue);">620 steps/hr</span>
              </div>
              <div class="gauge-bar-bg"><div id="bar-steps" class="gauge-bar-fill" style="width:65%; background:var(--accent-blue);"></div></div>
            </div>

            <div style="margin-bottom:0;">
              <div style="display:flex; justify-content:space-between; font-size:0.82rem;">
                <span>Circadian Start Hour Variance:</span>
                <span id="gauge-circadian" class="num" style="font-weight:700; color:var(--accent-purple);">0.6 hrs (Regular)</span>
              </div>
              <div class="gauge-bar-bg"><div id="bar-circadian" class="gauge-bar-fill" style="width:25%; background:var(--accent-purple);"></div></div>
            </div>
          </div>

          <!-- MODEL OUTPUTS -->
          <div class="card" style="background:var(--bg-secondary);">
            <div class="card-title">Model State & Predicted Severity</div>
            
            <div class="metric-pill">
              <span>Observed PHQ-9 Score:</span>
              <span id="traj-obs-score" class="num" style="font-weight:700; color:#fff;">4.0 pts</span>
            </div>
            <div class="metric-pill">
              <span>Model Predicted Score (s&#770;<sub>T</sub>):</span>
              <span id="traj-pred-score" class="num" style="font-weight:700; color:var(--accent-cyan);">4.8 pts</span>
            </div>
            <div class="metric-pill">
              <span>Predicted Severity Tier:</span>
              <span id="traj-pred-tier"><span class="badge badge-green">None-Minimal (0)</span></span>
            </div>
            <div class="metric-pill">
              <span>Within-Subject &Delta; (s<sub>T</sub> - s<sub>T-1</sub>):</span>
              <span id="traj-delta" class="num" style="font-weight:700;">-- (Baseline)</span>
            </div>
            <div class="metric-pill">
              <span>MCID Clinical Classification:</span>
              <span id="traj-mcid"><span class="badge badge-blue">Baseline Intake</span></span>
            </div>
          </div>

          <!-- CLINICIAN ACTION CARD -->
          <div class="card" style="background:var(--bg-secondary);">
            <div class="card-title">Clinical Decision Alert</div>
            
            <div id="alert-status-box" style="padding:14px; border-radius:8px; margin-bottom:12px; background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.3);">
              <div id="alert-title" style="font-weight:700; color:var(--accent-green); font-size:0.92rem;">NORMAL MONITORING</div>
              <div id="alert-sub" style="font-size:0.8rem; color:var(--text-secondary); margin-top:4px;">No severe depression risk detected. Operating threshold $T_4^* = 0.2100$ not breached.</div>
            </div>

            <div style="font-size:0.82rem; color:var(--text-secondary); line-height:1.5;">
              <strong style="color:#fff;">Biomarker Trajectory Insight:</strong><br>
              <span id="traj-clinical-note">Patient is maintaining stable physical activity and consistent sleep duration. Behavioral markers show no depressive relapse precursors.</span>
            </div>
          </div>

        </div>
      </div>

      <!-- INTERACTIVE SECTION 4: MCID THRESHOLD EXPLORER -->
      <div class="card" style="margin-top:24px;">
        <h3 style="font-size:1.1rem; margin-bottom:6px;">4. Minimal Clinically Important Difference (MCID) Metric Explorer</h3>
        <p style="color:var(--text-secondary); font-size:0.85rem; margin-bottom:16px;">
          According to validated psychiatric guidelines (Kroenke et al., 2001), a within-subject change of <strong>&plusmn;5 points on the PHQ-9</strong> represents a clinically meaningful difference. Test how the 3-class progression metric categorizes score shifts:
        </p>

        <div style="display:flex; align-items:center; gap:20px; background:var(--bg-secondary); padding:16px; border-radius:10px;">
          <div style="flex:1;">
            <div style="display:flex; justify-content:space-between; margin-bottom:6px; font-size:0.85rem;">
              <span>Simulated Score Change (&Delta;s = s<sub>t</sub> - s<sub>t-1</sub>):</span>
              <span id="mcid-val-label" class="num" style="font-size:1rem; font-weight:700; color:var(--accent-cyan);">+6 pts</span>
            </div>
            <input type="range" id="mcid-slider" min="-15" max="15" step="1" value="6" oninput="updateMCIDDemo(this.value)">
          </div>

          <div style="width:280px; text-align:center; padding:12px; border-radius:8px; background:var(--bg-card); border:1px solid var(--border-color);">
            <div style="font-size:0.75rem; color:var(--text-muted); text-transform:uppercase;">Progression Tier</div>
            <div id="mcid-badge-demo" style="margin-top:6px;"><span class="badge badge-rose" style="font-size:0.9rem; padding:6px 12px;">WORSENED (&ge; +5 pts)</span></div>
          </div>
        </div>
      </div>

    </div>

    <!-- TAB 3: BENCHMARKS & POLICIES -->
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

    <!-- TAB 4: THRESHOLDS -->
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
              <span id="slider-val-label" class="num" style="color:var(--accent-cyan); font-weight:700;">0.2100</span>
            </div>
            <input type="range" id="threshold-slider" min="0.02" max="0.50" step="0.01" value="0.21" oninput="updateThresholdSimulator(this.value)">
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
            <strong>Objective Function:</strong> $J(T_4) = \text{{Macro-F1}}(\text{{val}}) + 0.50 \times \text{{F1}}(\text{{val, Severe}})$<br>
            <strong>Mandatory Guardrails:</strong> Severe Recall &ge; 20.0% and Severe Precision &ge; 10.0%.<br><br>
            <strong>Key Clinical Finding:</strong> Default argmax yields only 25.0% severe recall (missing 54 out of 72 severe patients). 
            Optimizing the decision threshold to <strong>$T_4^* = 0.2100$</strong> boosts severe recall to <strong>41.7%</strong> (+66.7% relative gain), detecting 12 additional high-risk patients before acute crisis.
          </p>
        </div>
      </div>
    </div>

    <!-- TAB 5: CALIBRATION -->
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

    <!-- TAB 6: SHAP & EXPLANATIONS -->
    <div id="shap" class="tab-pane">
      
      <!-- HERO EXPLANATION: WHAT IS SHAP -->
      <div class="card" style="margin-bottom:24px; background:linear-gradient(135deg, rgba(24,34,56,0.95) 0%, rgba(19,28,46,0.95) 100%); border-color:rgba(59,130,246,0.3);">
        <span class="badge badge-purple" style="margin-bottom:8px;">Model Interpretability & Audit Trail</span>
        <h2 style="font-size:1.35rem; color:#fff; margin-bottom:8px;">TreeSHAP: The Mathematical "Receipt" Behind Every Decision</h2>
        <p style="color:var(--text-secondary); font-size:0.9rem; max-width:900px; line-height:1.6;">
          In complex machine learning models (like XGBoost with 150 features across hundreds of decision trees), clinicians cannot blindly trust a black-box probability. 
          <strong>SHAP (SHapley Additive exPlanations)</strong> calculates the exact fair mathematical contribution of each feature using cooperative game theory:
        </p>
        
        <div style="background:var(--bg-secondary); border:1px solid var(--border-color); border-radius:10px; padding:14px 18px; margin-top:16px; font-family:var(--font-mono); font-size:0.88rem; color:var(--text-primary); display:flex; align-items:center; justify-content:center; gap:12px; flex-wrap:wrap;">
          <span style="color:var(--text-muted);">Base Population Risk (4.6%)</span>
          <span style="color:var(--accent-cyan); font-weight:700;">+</span>
          <span style="color:var(--accent-rose); font-weight:600;">Positive Pushes (+ Risk Factors)</span>
          <span style="color:var(--accent-blue); font-weight:700;">-</span>
          <span style="color:var(--accent-green); font-weight:600;">Protective Buffers (- Resilience)</span>
          <span style="color:#fff; font-weight:700;">=</span>
          <span style="color:var(--accent-amber); font-weight:700;">Patient's Final Severe Risk</span>
        </div>
      </div>

      <!-- LOCAL PATIENT CASE STUDY WATERFALL EXPLORER -->
      <div class="card" style="margin-bottom:28px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; flex-wrap:wrap; gap:10px;">
          <div>
            <h3 style="font-size:1.15rem;">Interactive Patient Waterfall Explanation</h3>
            <p style="color:var(--text-secondary); font-size:0.85rem;">
              Select a clinical test case to inspect the exact feature pushes and pulls that dictated the model's decision:
            </p>
          </div>
          <div style="display:flex; gap:8px;">
            <button class="patient-pill active" onclick="selectShapCase('tp', this)">🚨 Case 1: True Positive Severe Alert</button>
            <button class="patient-pill" onclick="selectShapCase('fp', this)">⚠️ Case 2: False Positive Alert</button>
            <button class="patient-pill" onclick="selectShapCase('fn', this)">🔍 Case 3: False Negative (Missed)</button>
          </div>
        </div>

        <div class="grid-3" style="margin-bottom:18px;">
          <div class="metric-pill"><span>Evaluation Case Type:</span><span id="shap-case-type" class="num" style="font-weight:700;">--</span></div>
          <div class="metric-pill"><span>True Severity Tier:</span><span id="shap-case-true" class="num" style="font-weight:700;">--</span></div>
          <div class="metric-pill"><span>Model Prediction:</span><span id="shap-case-pred" class="num" style="font-weight:700;">--</span></div>
        </div>

        <!-- WATERFALL BARS CONTAINER -->
        <div style="background:var(--bg-secondary); border:1px solid var(--border-color); border-radius:10px; padding:16px;">
          <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:var(--text-muted); text-transform:uppercase; margin-bottom:12px; border-bottom:1px solid var(--border-color); padding-bottom:8px;">
            <span style="width:240px;">Feature & Domain</span>
            <span style="width:110px; text-align:center;">Observed Value</span>
            <span style="flex:1; text-align:center;">SHAP Impact Direction (Left: Reduces Risk | Right: Increases Risk)</span>
            <span style="width:90px; text-align:right;">Impact (&Delta;p)</span>
          </div>
          <div id="shap-waterfall-rows">
            <!-- Injected via JavaScript -->
          </div>
        </div>

        <div id="shap-case-narrative" style="margin-top:14px; background:rgba(59,130,246,0.08); border:1px solid rgba(59,130,246,0.3); border-radius:8px; padding:14px; font-size:0.85rem; color:var(--text-secondary); line-height:1.5;">
          <!-- Injected via JavaScript -->
        </div>
      </div>

      <!-- GLOBAL BENCHMARKS & DOMAINS -->
      <div class="grid-2">
        <div class="card">
          <h3 style="font-size:1.1rem; margin-bottom:12px;">Global Population Importance (Top 15 Biomarkers)</h3>
          <p style="font-size:0.82rem; color:var(--text-secondary); margin-bottom:10px;">Mean absolute impact across all classes (E[|&phi;|]) over the entire cohort.</p>
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
          <p style="font-size:0.82rem; color:var(--text-secondary); margin-bottom:10px;">Relative information share derived from passive sensors vs. clinical history.</p>
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

      selectShapCase('tp', null);
    }}

    function selectShapCase(caseKey, btnElem) {{
      if (btnElem) {{
        document.querySelectorAll('#shap .patient-pill').forEach(b => b.classList.remove('active'));
        btnElem.classList.add('active');
      }}

      const localCases = (STAGE5_DATA.shap && (STAGE5_DATA.shap.local_cases || STAGE5_DATA.shap.local_explanations));
      if (!localCases) return;
      const keyMap = {{
        'tp': 'true_positive_severe_alert',
        'fp': 'false_positive_severe_alert',
        'fn': 'false_negative_missed_severe'
      }};
      const caseData = localCases[keyMap[caseKey]];
      if (!caseData) return;

      document.getElementById('shap-case-type').innerText = caseData.case_type;
      document.getElementById('shap-case-true').innerHTML = `<span class="badge ${{caseData.true_severity_tier === 4 ? 'badge-rose' : 'badge-blue'}}">Tier ${{caseData.true_severity_tier}} (${{caseData.true_severity_tier === 4 ? 'Severe' : (caseData.true_severity_tier === 1 ? 'Mild' : 'Moderate')}})</span>`;
      document.getElementById('shap-case-pred').innerHTML = `<span class="badge ${{caseData.predicted_severity_tier === 4 ? 'badge-rose' : 'badge-green'}}">Tier ${{caseData.predicted_severity_tier}} (${{caseData.predicted_severity_tier === 4 ? 'Severe' : 'Mild'}})</span>`;

      const container = document.getElementById('shap-waterfall-rows');
      container.innerHTML = '';

      const features = caseData.top_10_driving_features || [];
      let maxImpact = 0.3;
      features.forEach(f => {{
        if (Math.abs(f.shap_impact_on_severe) > maxImpact) maxImpact = Math.abs(f.shap_impact_on_severe);
      }});

      features.forEach(f => {{
        const row = document.createElement('div');
        row.style.cssText = 'display:flex; align-items:center; justify-content:space-between; padding:8px 0; border-bottom:1px solid rgba(35, 49, 79, 0.4); font-size:0.83rem;';

        const isPositive = f.shap_impact_on_severe >= 0;
        const barWidthPct = Math.min(48, (Math.abs(f.shap_impact_on_severe) / maxImpact) * 48);

        row.innerHTML = `
          <div style="width:240px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
            <code>${{f.feature}}</code>
            <div style="font-size:0.75rem; color:var(--text-muted);">${{f.domain}}</div>
          </div>
          <div style="width:110px; text-align:center;" class="num">
            ${{f.feature_value_scaled > 0 ? '+' : ''}}${{f.feature_value_scaled.toFixed(2)}}&sigma;
          </div>
          <div style="flex:1; display:flex; align-items:center; height:22px; position:relative; margin:0 14px; background:rgba(10,14,23,0.6); border-radius:4px;">
            <div style="position:absolute; left:50%; top:0; bottom:0; width:2px; background:var(--border-color); z-index:2;"></div>
            ${{isPositive ? `
              <div style="position:absolute; left:50%; width:${{barWidthPct}}%; height:14px; background:rgba(244,63,94,0.75); border-radius:0 3px 3px 0; box-shadow:0 0 6px rgba(244,63,94,0.4);" title="+${{f.shap_impact_on_severe.toFixed(4)}} pushes risk HIGHER"></div>
            ` : `
              <div style="position:absolute; right:50%; width:${{barWidthPct}}%; height:14px; background:rgba(59,130,246,0.75); border-radius:3px 0 0 3px; box-shadow:0 0 6px rgba(59,130,246,0.4);" title="${{f.shap_impact_on_severe.toFixed(4)}} pushes risk LOWER"></div>
            `}}
          </div>
          <div style="width:90px; text-align:right; font-weight:700;" class="num">
            <span style="color:${{isPositive ? 'var(--accent-rose)' : 'var(--accent-blue)'}};">${{isPositive ? '+' : ''}}${{f.shap_impact_on_severe.toFixed(4)}}</span>
          </div>
        `;
        container.appendChild(row);
      }});

      const narrativeBox = document.getElementById('shap-case-narrative');
      if (caseKey === 'tp') {{
        narrativeBox.innerHTML = `<strong>Clinical Audit Narrative (True Positive):</strong> The model accurately triggered a severe depression alert. As shown in the red bars, acute financial stress (+0.6438) and comorbid chronic neuropathic pain (+0.1256) heavily increased predicted risk, compounded by nocturnal sleep awakenings (+0.0737). The clinician can confirm behavioral and physical vulnerability.`;
      }} else if (caseKey === 'fp') {{
        narrativeBox.innerHTML = `<strong>Clinical Audit Narrative (False Positive):</strong> The patient was clinically Mild (Tier 1) but flagged as Severe by the model. The explanation shows that acute physical comorbidities and temporary sleep disruption pushed the model into an over-alert state. In practice, this prompt leads to a nurse check-in without medication escalation.`;
      }} else {{
        narrativeBox.innerHTML = `<strong>Clinical Audit Narrative (False Negative):</strong> The patient had Severe Depression (Tier 4), but the model under-predicted (Tier 1). The blue bars show that high educational attainment (-0.5585) and financial stability (-0.1674) acted as strong protective buffers in the model's eyes, masking severe emotional suffering. This highlights why decision thresholds must be tuned to T_4* = 0.2100.`;
      }}
    }}

    function updateThresholdSimulator(val) {{
      const t = parseFloat(val);
      document.getElementById('slider-val-label').innerText = t.toFixed(4);

      if (!STAGE5_DATA.thresholds || !STAGE5_DATA.thresholds.sweep_curve) return;
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

    // =========================================================================
    // LONGITUDINAL INTERACTIVE FUNCTIONS
    // =========================================================================

    function setWindowDemo(w, btnElem) {{
      document.querySelectorAll('#longitudinal .patient-pill').forEach(b => {{
        if (b.innerText.includes('Window') || b.innerText.includes('History')) b.classList.remove('active');
      }});
      if (btnElem) btnElem.classList.add('active');

      const targetPeriod = 6;
      const container = document.getElementById('window-cells-container');
      container.innerHTML = '';

      let startP = 1;
      if (w === 1) startP = 5;
      else if (w === 2) startP = 4;
      else if (w === 3) startP = 3;
      else if (w === 'full') startP = 1;

      for (let p = 1; p <= 12; p++) {{
        const cell = document.createElement('span');
        cell.className = 'period-cell';
        cell.innerText = 'P' + p;

        if (p === targetPeriod) {{
          cell.classList.add('target');
          cell.title = 'Period ' + p + ' (Current Target Assessment)';
        }} else if (p >= startP && p < targetPeriod) {{
          cell.classList.add('in-window');
          cell.title = 'Period ' + p + ' (Preceding History in Window)';
        }} else if (p > targetPeriod) {{
          cell.classList.add('quarantined');
          cell.title = 'Period ' + p + ' (QUARANTINED FUTURE: Zero Leakage)';
        }}
        container.appendChild(cell);
      }}

      const expl = document.getElementById('window-expl-text');
      if (w === 1) {{
        expl.innerHTML = '<strong>W = 1 Configuration:</strong> Uses immediate preceding Period 5 sensor data (H_T of shape 1 &times; 150) plus Period 6 current sensors to predict PHQ-9 at Month 6. Prospectively selected configuration on validation Macro-F1 (0.3359).';
      }} else if (w === 2) {{
        expl.innerHTML = '<strong>W = 2 Configuration:</strong> Uses 2 preceding periods (Periods 4 and 5) to capture 2-month sequential trajectory (H_T of shape 2 &times; 150).';
      }} else if (w === 3) {{
        expl.innerHTML = '<strong>W = 3 Configuration:</strong> Uses 3 preceding periods (Periods 3, 4, and 5) representing a full prior quarter of behavioral history.';
      }} else {{
        expl.innerHTML = '<strong>Full Cumulative History:</strong> Right-aligned, variable-length history from baseline up to Period 5, padded with masking for missing periods.';
      }}
    }}

    function updateMCIDDemo(val) {{
      const delta = parseInt(val);
      document.getElementById('mcid-val-label').innerText = (delta > 0 ? '+' : '') + delta + ' pts';
      const badge = document.getElementById('mcid-badge-demo');

      if (delta <= -5) {{
        badge.innerHTML = '<span class="badge badge-green" style="font-size:0.9rem; padding:6px 12px;">IMPROVED (&Delta; &le; -5 pts)</span>';
      }} else if (delta >= 5) {{
        badge.innerHTML = '<span class="badge badge-rose" style="font-size:0.9rem; padding:6px 12px;">WORSENED (&ge; +5 pts)</span>';
      }} else {{
        badge.innerHTML = '<span class="badge badge-blue" style="font-size:0.9rem; padding:6px 12px;">STABLE (-4 .. +4 pts)</span>';
      }}
    }}

    // Patient Simulator Trajectory Data
    const TRAJECTORIES_DB = {{
      relapse: {{
        name: "Patient #1062",
        type: "Worsening / Relapse Course",
        quarters: [
          {{ q: "Q1", p: 3, obs: 4, pred: 4.8, sleep: 7.4, sed: 7.8, steps: 620, circ: "0.6 hrs (Regular)", tier: "None-Minimal (0)", tierClass: "badge-green", delta: "-- (Baseline)", mcid: "Baseline", alert: false, note: "Patient is maintaining stable physical activity and consistent sleep duration. Behavioral markers show no depressive relapse precursors." }},
          {{ q: "Q2", p: 6, obs: 5, pred: 6.1, sleep: 6.8, sed: 8.6, steps: 540, circ: "1.1 hrs (Mild shift)", tier: "Mild (1)", tierClass: "badge-blue", delta: "+1.3 pts", mcid: "Stable (-4 .. +4)", alert: false, note: "Slight decrease in total sleep duration and modest rise in sedentary hours. Patient remains within subclinical bounds." }},
          {{ q: "Q3", p: 9, obs: 8, pred: 8.4, sleep: 5.7, sed: 11.2, steps: 380, circ: "2.3 hrs (Disrupted)", tier: "Mild (1) ➜ Escalating", tierClass: "badge-amber", delta: "+2.3 pts", mcid: "Stable / Warning", alert: false, note: "Marked sleep fragmentation and waking hour lethargy detected by actigraphy. Sensor metrics indicate acute behavioral disruption." }},
          {{ q: "Q4", p: 12, obs: 14, pred: 12.9, sleep: 4.8, sed: 13.8, steps: 210, circ: "3.4 hrs (Circadian Misalignment)", tier: "Moderate (2)", tierClass: "badge-rose", delta: "+4.5 pts (Borderline MCID)", mcid: "Worsening Risk Alert", alert: true, note: "CRITICAL ALERT: Sleep severely reduced (<5 hrs), sedentary time surged to nearly 14 hrs/day. Model triggered severe decision alert (T_4* = 0.2100) weeks before formal clinical assessment." }}
        ]
      }},
      remission: {{
        name: "Patient #1574",
        type: "Treatment Remission Course",
        quarters: [
          {{ q: "Q1", p: 3, obs: 16, pred: 14.2, sleep: 5.1, sed: 13.2, steps: 240, circ: "3.1 hrs (Disrupted)", tier: "Moderately Severe (3)", tierClass: "badge-rose", delta: "-- (Baseline)", mcid: "Baseline Intake", alert: true, note: "High severity intake. Severe psychomotor retardation and nocturnal fragmentation documented on sensors." }},
          {{ q: "Q2", p: 6, obs: 13, pred: 12.1, sleep: 6.0, sed: 11.5, steps: 360, circ: "2.0 hrs (Stabilizing)", tier: "Moderate (2)", tierClass: "badge-amber", delta: "-2.1 pts", mcid: "Stable", alert: false, note: "Therapeutic intervention underway. Sleep duration beginning to rebound, sedentary hours slowly declining." }},
          {{ q: "Q3", p: 9, obs: 9, pred: 9.8, sleep: 6.9, sed: 9.4, steps: 510, circ: "1.2 hrs (Consistent)", tier: "Mild (1)", tierClass: "badge-blue", delta: "-4.4 pts", mcid: "Response Emerging", alert: false, note: "Robust behavioral activation. Hourly physical activity increased by >40% relative to baseline intake." }},
          {{ q: "Q4", p: 12, obs: 6, pred: 7.4, sleep: 7.5, sed: 8.0, steps: 640, circ: "0.7 hrs (Normal)", tier: "Mild (1) ➜ Subclinical", tierClass: "badge-green", delta: "-6.8 pts (MCID Met)", mcid: "IMPROVED (Delta <= -5 pts)", alert: false, note: "Full clinical remission confirmed. Total within-subject reduction of -10 observed points (-6.8 predicted points), safely exceeding MCID improvement cutoff." }}
        ]
      }},
      stable: {{
        name: "Patient #1026",
        type: "Outpatient Stable Course",
        quarters: [
          {{ q: "Q1", p: 3, obs: 3, pred: 3.2, sleep: 7.8, sed: 7.2, steps: 680, circ: "0.5 hrs (Regular)", tier: "None-Minimal (0)", tierClass: "badge-green", delta: "-- (Baseline)", mcid: "Baseline", alert: false, note: "Healthy outpatient baseline. Circadian stability and physical activity within normal adult reference intervals." }},
          {{ q: "Q2", p: 6, obs: 2, pred: 2.8, sleep: 7.6, sed: 7.4, steps: 660, circ: "0.6 hrs (Regular)", tier: "None-Minimal (0)", tierClass: "badge-green", delta: "-0.4 pts", mcid: "Stable", alert: false, note: "Stable sleep-wake cycles. Zero clinical or behavioral elevation." }},
          {{ q: "Q3", p: 9, obs: 3, pred: 3.1, sleep: 7.7, sed: 7.1, steps: 690, circ: "0.5 hrs (Regular)", tier: "None-Minimal (0)", tierClass: "badge-green", delta: "+0.3 pts", mcid: "Stable", alert: false, note: "Continuous sensor monitoring confirms uninterrupted longitudinal stability across all modalities." }},
          {{ q: "Q4", p: 12, obs: 2, pred: 2.9, sleep: 7.8, sed: 7.3, steps: 670, circ: "0.6 hrs (Regular)", tier: "None-Minimal (0)", tierClass: "badge-green", delta: "-0.2 pts", mcid: "Stable", alert: false, note: "Completed 12-month passive observation with zero relapse events or significant variance." }}
        ]
      }}
    }};

    let currentPatientKey = 'relapse';
    let currentQuarterIdx = 0;

    function selectPatientTrajectory(key, btnElem) {{
      currentPatientKey = key;
      document.querySelectorAll('#longitudinal .patient-pill').forEach(b => {{
        if (b.innerText.includes('Patient')) b.classList.remove('active');
      }});
      if (btnElem) btnElem.classList.add('active');

      const pat = TRAJECTORIES_DB[key];
      // Update scrubber labels
      document.getElementById('scrub-q1-score').innerText = 'PHQ-9: ' + pat.quarters[0].obs + ' pts';
      document.getElementById('scrub-q2-score').innerText = 'PHQ-9: ' + pat.quarters[1].obs + ' pts';
      document.getElementById('scrub-q3-score').innerText = 'PHQ-9: ' + pat.quarters[2].obs + ' pts';
      document.getElementById('scrub-q4-score').innerText = 'PHQ-9: ' + pat.quarters[3].obs + ' pts';

      scrubQuarter(0, document.querySelectorAll('.scrub-btn')[0]);
    }}

    function scrubQuarter(idx, btnElem) {{
      currentQuarterIdx = idx;
      document.querySelectorAll('.scrub-btn').forEach(b => b.classList.remove('active'));
      if (btnElem) btnElem.classList.add('active');

      const pat = TRAJECTORIES_DB[currentPatientKey];
      const qData = pat.quarters[idx];

      // Update gauges
      document.getElementById('gauge-sleep').innerText = qData.sleep.toFixed(1) + ' hrs';
      document.getElementById('bar-sleep').style.width = Math.min(100, (qData.sleep / 9.0) * 100) + '%';

      document.getElementById('gauge-sedentary').innerText = qData.sed.toFixed(1) + ' hrs';
      document.getElementById('bar-sedentary').style.width = Math.min(100, (qData.sed / 16.0) * 100) + '%';

      document.getElementById('gauge-steps').innerText = qData.steps + ' steps/hr';
      document.getElementById('bar-steps').style.width = Math.min(100, (qData.steps / 800) * 100) + '%';

      document.getElementById('gauge-circadian').innerText = qData.circ;
      document.getElementById('bar-circadian').style.width = (qData.circ.includes('Regular') ? 25 : (qData.circ.includes('Disrupted') ? 70 : 90)) + '%';

      // Update model outputs
      document.getElementById('traj-obs-score').innerText = qData.obs + ' pts';
      document.getElementById('traj-pred-score').innerText = qData.pred.toFixed(1) + ' pts';
      document.getElementById('traj-pred-tier').innerHTML = `<span class="badge ${{qData.tierClass}}">${{qData.tier}}</span>`;
      document.getElementById('traj-delta').innerText = qData.delta;
      document.getElementById('traj-mcid').innerHTML = `<span class="badge ${{qData.mcid.includes('IMPROVED') ? 'badge-green' : (qData.mcid.includes('Worsening') ? 'badge-rose' : 'badge-blue')}}">${{qData.mcid}}</span>`;

      // Update Alert Box
      const alertBox = document.getElementById('alert-status-box');
      const alertTitle = document.getElementById('alert-title');
      const alertSub = document.getElementById('alert-sub');

      if (qData.alert) {{
        alertBox.style.background = 'rgba(244,63,94,0.15)';
        alertBox.style.borderColor = 'rgba(244,63,94,0.5)';
        alertTitle.style.color = 'var(--accent-rose)';
        alertTitle.innerText = '🚨 HIGH RISK DECISION ALERT TRIGGERED';
        alertSub.innerText = 'Post-hoc threshold T_4* = 0.2100 breached. Clinician-in-the-loop triage alert dispatched.';
      }} else {{
        alertBox.style.background = 'rgba(16,185,129,0.1)';
        alertBox.style.borderColor = 'rgba(16,185,129,0.3)';
        alertTitle.style.color = 'var(--accent-green)';
        alertTitle.innerText = 'NORMAL MONITORING STATUS';
        alertSub.innerText = 'Behavioral sensors remain within expected confidence intervals. No acute escalation detected.';
      }}

      document.getElementById('traj-clinical-note').innerText = qData.note;
    }}

    // Init on load
    window.addEventListener('DOMContentLoaded', () => {{
      populateBenchmarks();
      populateCalibration();
      populateSHAP();
      setWindowDemo(1, null);
      updateMCIDDemo(6);
      selectPatientTrajectory('relapse', null);

      if (STAGE5_DATA.thresholds && STAGE5_DATA.thresholds.selected_threshold) {{
        document.getElementById('threshold-slider').value = STAGE5_DATA.thresholds.selected_threshold;
        updateThresholdSimulator(STAGE5_DATA.thresholds.selected_threshold);
      }}
    }});
  </script>
</body>
</html>
"""
    # Write to dashboard/index.html
    main_html_path = dashboard_dir / "index.html"
    with open(main_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    # Write to top-level clinical_decision_dashboard.html for convenient access
    root_html_path = output_dir / "clinical_decision_dashboard.html"
    with open(root_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"Dashboard created successfully at: {main_html_path}")
    print(f"Dashboard alias created at: {root_html_path}")
    return main_html_path
