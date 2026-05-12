
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest


np.random.seed(42)
dates     = pd.date_range(end=datetime.today(), periods=30, freq="D")
revenue   = np.random.normal(85_000, 15_000, 30).clip(40_000, 140_000)
anomalies = np.random.poisson(12, 30)


DATE_RANGE_LABEL = f"{dates[0].strftime('%b %d')} – {dates[-1].strftime('%b %d, %Y')}"


np.random.seed(42)
_n = 5000
_train_X = np.column_stack([
    np.random.exponential(150, _n),          # transaction_amount
    np.random.poisson(2, _n),                # order_velocity
    np.random.randint(30, 3650, _n),         # account_age_days
    np.random.beta(1, 5, _n),               # ip_risk_score
    np.random.randint(0, 6, _n),            # payment_method (encoded)
])
_IF_MODEL = IsolationForest(n_estimators=200, contamination=0.075, random_state=42)
_IF_MODEL.fit(_train_X)
_PAYMENT_ENC = {"Credit Card":0,"Debit Card":1,"PayPal":2,"UPI":3,"Crypto":4,"COD":5}

score_bins   = [f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)]
score_counts = [8200,12400,10800,7200,4100,2300,1400,980,620,290]
score_colors = ["#00e5a0"]*4 + ["#f5a623"]*2 + ["#ff4e4e"]*4

categories = ["Electronics","Apparel","Home","Beauty","Sports","Accessories"]
cat_aov    = [320, 85, 140, 65, 110, 45]
cat_fraud  = [3.1, 0.9, 1.4, 0.6, 1.1, 0.4]
cat_vol    = [900, 700, 600, 500, 650, 400]

fpr = [0,0.02,0.05,0.08,0.12,0.18,0.25,0.35,0.5,0.65,0.8,1.0]
tpr = [0,0.28,0.52,0.65,0.74,0.81,0.86,0.90,0.93,0.95,0.97,1.0]

seg_labels = ["Low Risk","Pre-Fraud","High Risk"]
seg_values = [38500, 6200, 3612]
seg_colors = ["#00e5a0","#f5a623","#ff4e4e"]

alerts = [
    {"id":"TXN-9841","amt":"$3,420","note":"3 cards, 2 IPs, Electronics","level":"HIGH"},
    {"id":"TXN-9817","amt":"$7,800","note":"Velocity spike, 11 orders/hr","level":"HIGH"},
    {"id":"TXN-9803","amt":"$890", "note":"New device, billing mismatch","level":"WARN"},
    {"id":"TXN-9799","amt":"$430", "note":"Promo abuse pattern","level":"WARN"},
    {"id":"TXN-9788","amt":"$210", "note":"Manually reviewed, cleared","level":"SAFE"},
]

shap_features = ["Transaction amount","Order velocity (1h)","Device fingerprint",
                 "Billing-shipping match","Time since registration",
                 "IP geolocation risk","Promo code usage","Category risk score"]
shap_vals = [0.91,0.78,0.65,0.58,0.44,0.39,0.29,0.21]

channels  = ["Organic","Paid Ads","Referral","Social","Email","Direct"]
low_r  = [4200,2100,1800,1400,2600,3100]
pre_f  = [310, 480, 290, 620, 380, 210]
high_r = [80,  210, 60,  290, 140, 70]

models    = ["Logistic Reg","Decision Tree","Random Forest","SVM","Isolation Forest","XGBoost"]
acc_vals  = [0.81, 0.84, 0.93, 0.89, 0.91, 0.94]
prec_vals = [0.78, 0.80, 0.91, 0.87, 0.88, 0.93]
rec_vals  = [0.75, 0.82, 0.89, 0.85, 0.86, 0.92]
f1_vals   = [0.76, 0.81, 0.90, 0.86, 0.87, 0.92]


BG_MAIN    = "#0d1117"
BG_SIDEBAR = "#0a0e14"
BG_CARD    = "#161b23"
BG_CARD2   = "#1a2030"
ACCENT1    = "#00e5a0"
ACCENT2    = "#f5a623"
ACCENT3    = "#ff4e4e"
ACCENT4    = "#4e9eff"
TEXT_PRI   = "#e8edf3"
TEXT_SEC   = "#7a8799"
BORDER     = "#222d3a"
PLOT_BG    = "rgba(0,0,0,0)"
GRID_C     = "#1e2a38"


def base_layout(fig, h=260):
    fig.update_layout(
        height=h, paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=dict(family="monospace", color=TEXT_SEC, size=11),
        margin=dict(l=8, r=8, t=8, b=30),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    )
    return fig

def fig_trend():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=revenue/1000, name="Revenue ($k)", yaxis="y1",
        line=dict(color=ACCENT4, width=2), fill="tozeroy",
        fillcolor="rgba(78,158,255,0.07)", mode="lines"))
    fig.add_trace(go.Scatter(x=dates, y=anomalies, name="Anomalies", yaxis="y2",
        line=dict(color=ACCENT3, width=2, dash="dot"), mode="lines"))
    fig.update_layout(
        yaxis=dict(title="Revenue ($k)", color=ACCENT4, gridcolor=GRID_C, zeroline=False),
        yaxis2=dict(title="Anomalies", overlaying="y", side="right", color=ACCENT3, gridcolor=GRID_C, zeroline=False),
        xaxis=dict(gridcolor=GRID_C, tickformat="%b %d"),
        legend=dict(orientation="h", y=1.12, x=0))
    return base_layout(fig)

def fig_histogram():
    fig = go.Figure(go.Bar(x=score_bins, y=score_counts,
        marker_color=score_colors, marker_line_width=0))
    fig.update_layout(
        xaxis=dict(title="Anomaly Score", gridcolor=GRID_C),
        yaxis=dict(title="Transactions", gridcolor=GRID_C, zeroline=False), bargap=0.1)
    return base_layout(fig)

def fig_bubble():
    fig = go.Figure()
    cols = [ACCENT3, ACCENT4, ACCENT1, ACCENT2, "#b97fff", TEXT_SEC]
    for i, cat in enumerate(categories):
        fig.add_trace(go.Scatter(
            x=[cat_aov[i]], y=[cat_fraud[i]], mode="markers+text", name=cat,
            text=[cat], textposition="top center",
            textfont=dict(size=9, color=TEXT_PRI),
            marker=dict(size=cat_vol[i]/30, color=cols[i], opacity=0.75,
                        line=dict(width=1, color="rgba(255,255,255,0.15)"))))
    fig.update_layout(
        xaxis=dict(title="Avg Order Value ($)", gridcolor=GRID_C, zeroline=False, range=[0,380]),
        yaxis=dict(title="Fraud Rate (%)", gridcolor=GRID_C, zeroline=False, range=[0,4]),
        showlegend=False)
    return base_layout(fig)

def fig_roc():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name="Isolation Forest AUC 0.91",
        line=dict(color=ACCENT1, width=2.5), fill="tozeroy",
        fillcolor="rgba(0,229,160,0.07)", mode="lines"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Random baseline",
        line=dict(color=TEXT_SEC, width=1.2, dash="dash"), mode="lines"))
    fig.update_layout(
        xaxis=dict(title="False Positive Rate", gridcolor=GRID_C, zeroline=False),
        yaxis=dict(title="True Positive Rate", gridcolor=GRID_C, zeroline=False),
        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=9)))
    return base_layout(fig)

def fig_donut():
    fig = go.Figure(go.Pie(
        labels=seg_labels, values=seg_values, hole=0.62,
        marker=dict(colors=seg_colors, line=dict(color=BG_CARD, width=3)),
        textinfo="none", sort=False))
    fig.update_layout(showlegend=False,
        annotations=[dict(text="48,312", x=0.5, y=0.5, showarrow=False,
                          font=dict(color=TEXT_PRI, size=16))])
    return base_layout(fig, 220)

def fig_shap():
    cols = [ACCENT3 if v>0.6 else ACCENT2 if v>0.4 else ACCENT4 for v in shap_vals]
    fig = go.Figure(go.Bar(
        x=shap_vals, y=shap_features, orientation="h",
        marker_color=cols, marker_line_width=0,
        text=[f"{v:.2f}" for v in shap_vals],
        textposition="outside", textfont=dict(color=TEXT_SEC, size=10)))
    fig.update_layout(
        xaxis=dict(range=[0,1.15], gridcolor=GRID_C, zeroline=False),
        yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"), bargap=0.25)
    return base_layout(fig, 280)

def fig_segment():
    fig = go.Figure()
    for data, name, color in [(low_r,"Low Risk",ACCENT1),(pre_f,"Pre-Fraud",ACCENT2),(high_r,"High Risk",ACCENT3)]:
        fig.add_trace(go.Bar(x=channels, y=data, name=name, marker_color=color, marker_line_width=0))
    fig.update_layout(barmode="stack",
        xaxis=dict(gridcolor=GRID_C),
        yaxis=dict(gridcolor=GRID_C, zeroline=False),
        legend=dict(orientation="h", y=1.12, x=0))
    return base_layout(fig, 260)

def fig_model_bench():
    fig = go.Figure()
    for vals, name, color in [
        (acc_vals,"Accuracy",ACCENT1),(prec_vals,"Precision",ACCENT4),
        (rec_vals,"Recall",ACCENT2),(f1_vals,"F1-Score",ACCENT3)]:
        fig.add_trace(go.Bar(name=name, x=models, y=vals, marker_color=color, marker_line_width=0))
    fig.update_layout(barmode="group",
        xaxis=dict(gridcolor=GRID_C),
        yaxis=dict(gridcolor=GRID_C, zeroline=False, range=[0.6,1.0]),
        legend=dict(orientation="h", y=1.12, x=0))
    return base_layout(fig, 320)

def fig_confusion():
    z = [[9812, 188], [142, 9858]]
    fig = go.Figure(go.Heatmap(
        z=z, x=["Predicted Normal","Predicted Fraud"],
        y=["Actual Normal","Actual Fraud"],
        colorscale=[[0,"#0a0e14"],[0.5,"#003d29"],[1.0,ACCENT1]],
        text=[[str(v) for v in row] for row in z],
        texttemplate="%{text}", textfont=dict(size=20, color=TEXT_PRI),
        showscale=False))
    fig.update_layout(xaxis=dict(side="top"), yaxis=dict(autorange="reversed"))
    return base_layout(fig, 280)

def fig_daily_orders():
    orders  = np.random.poisson(1600, 30)
    returns = np.random.poisson(80, 30)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=dates, y=orders,  name="Orders",  marker_color=ACCENT4, marker_line_width=0))
    fig.add_trace(go.Bar(x=dates, y=returns, name="Returns", marker_color=ACCENT2, marker_line_width=0))
    fig.update_layout(barmode="stack",
        xaxis=dict(gridcolor=GRID_C, tickformat="%b %d"),
        yaxis=dict(gridcolor=GRID_C, zeroline=False),
        legend=dict(orientation="h", y=1.12))
    return base_layout(fig, 240)

def fig_payment_methods():
    methods     = ["Credit Card","Debit Card","PayPal","UPI","Crypto","COD"]
    fraud_by    = [2.1, 0.9, 0.4, 0.2, 4.8, 0.1]
    fig = go.Figure(go.Bar(
        x=methods, y=fraud_by,
        marker_color=[ACCENT3 if v>2 else ACCENT2 if v>1 else ACCENT1 for v in fraud_by],
        marker_line_width=0,
        text=[f"{v}%" for v in fraud_by], textposition="outside",
        textfont=dict(color=TEXT_SEC)))
    fig.update_layout(
        yaxis=dict(title="Fraud Rate (%)", gridcolor=GRID_C, zeroline=False),
        xaxis=dict(gridcolor=GRID_C))
    return base_layout(fig, 260)

def fig_corr_matrix():
    corr_labels = ["txn_amt","velocity","acct_age","ip_risk","promo","ship_match","label"]
    n = len(corr_labels)
    np.random.seed(7)
    corr = np.eye(n)
    corr[0][6]=corr[6][0]=0.68; corr[1][6]=corr[6][1]=0.72
    corr[3][6]=corr[6][3]=0.61; corr[4][6]=corr[6][4]=0.33
    corr[5][6]=corr[6][5]=-0.41
    for i in range(n):
        for j in range(i+1,n):
            if corr[i][j]==0:
                v = round(np.random.uniform(-0.3,0.3),2)
                corr[i][j]=corr[j][i]=v
    fig = go.Figure(go.Heatmap(
        z=corr, x=corr_labels, y=corr_labels,
        colorscale=[[0,ACCENT3],[0.5,BG_CARD2],[1,ACCENT1]],
        zmin=-1, zmax=1,
        text=[[f"{corr[i][j]:.2f}" for j in range(n)] for i in range(n)],
        texttemplate="%{text}", textfont=dict(size=9)))
    return base_layout(fig, 320)

def fig_amount_dist():
    # Use full dataset proportions: 44,700 legitimate + 3,612 fraud to correctly
    # reflect the true 7.48% class imbalance in density shapes
    np.random.seed(42)
    tx_fraud = np.random.exponential(400, 3612)
    tx_legit = np.random.exponential(150, 44700)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=tx_legit, name="Legitimate", marker_color=ACCENT1,
        opacity=0.7, nbinsx=50, histnorm="probability density"))
    fig.add_trace(go.Histogram(x=tx_fraud, name="Fraudulent", marker_color=ACCENT3,
        opacity=0.7, nbinsx=50, histnorm="probability density"))
    fig.update_layout(barmode="overlay",
        xaxis=dict(title="Transaction Amount ($)", gridcolor=GRID_C, range=[0,2000]),
        yaxis=dict(title="Density", gridcolor=GRID_C, zeroline=False),
        legend=dict(orientation="h", y=1.12))
    return base_layout(fig, 280)


def metric_card(label, value, delta, color=ACCENT1, icon="*"):
    return html.Div([
        html.Div(icon,  style={"fontSize":"20px","color":color,"marginBottom":"8px"}),
        html.Div(value, style={"fontSize":"26px","fontWeight":"700","color":TEXT_PRI,"letterSpacing":"-0.5px"}),
        html.Div(label, style={"fontSize":"11px","color":TEXT_SEC,"marginTop":"4px","textTransform":"uppercase","letterSpacing":"1px"}),
        html.Div(delta, style={"fontSize":"11px","color":color,"marginTop":"6px"}),
    ], style={"background":BG_CARD,"border":f"1px solid {BORDER}","borderTop":f"2px solid {color}",
              "borderRadius":"8px","padding":"20px 18px","flex":"1","minWidth":"0"})

def card(children, title=None, extra_style=None):
    s = {"background":BG_CARD,"border":f"1px solid {BORDER}","borderRadius":"8px","padding":"18px"}
    if extra_style:
        s.update(extra_style)
    inner = []
    if title:
        inner.append(html.Div(title, style={"fontSize":"10px","fontWeight":"600","color":TEXT_SEC,
            "letterSpacing":"2px","textTransform":"uppercase","marginBottom":"12px"}))
    inner += children if isinstance(children, list) else [children]
    return html.Div(inner, style=s)

def alert_row(info):
    dc  = {"HIGH":ACCENT3,"WARN":ACCENT2,"SAFE":ACCENT1}[info["level"]]
    bb  = {"HIGH":"rgba(255,78,78,0.15)","WARN":"rgba(245,166,35,0.15)","SAFE":"rgba(0,229,160,0.15)"}[info["level"]]
    bt  = {"HIGH":ACCENT3,"WARN":ACCENT2,"SAFE":ACCENT1}[info["level"]]
    lbl = {"HIGH":"High Risk","WARN":"Suspicious","SAFE":"Cleared"}[info["level"]]
    return html.Div([
        html.Div(style={"width":"8px","height":"8px","borderRadius":"50%","background":dc,
                        "flexShrink":"0","marginTop":"5px"}),
        html.Div([
            html.Div(f"{info['id']} - {info['amt']}", style={"fontSize":"13px","fontWeight":"600","color":TEXT_PRI}),
            html.Div(info["note"], style={"fontSize":"11px","color":TEXT_SEC,"marginTop":"2px"}),
        ], style={"flex":"1"}),
        html.Div(lbl, style={"background":bb,"color":bt,"padding":"3px 10px",
                             "borderRadius":"4px","fontSize":"10px","fontWeight":"600"}),
    ], style={"display":"flex","alignItems":"flex-start","gap":"12px","padding":"10px 0",
              "borderBottom":f"1px solid {BORDER}"})

def section_heading(title, subtitle=""):
    kids = [html.Div(title, style={"fontSize":"22px","fontWeight":"800","color":TEXT_PRI})]
    if subtitle:
        kids.append(html.Div(subtitle, style={"fontSize":"11px","color":TEXT_SEC,"marginTop":"3px"}))
    return html.Div(kids, style={"marginBottom":"22px"})


def page_overview():
    return html.Div([
        section_heading("System Integrity Dashboard",
            "Real-time e-commerce fraud detection - Isolation Forest - Synthetic dataset"),
        html.Div([
            metric_card("Total Revenue (30d)", "$2.84M", "Up 11.2% vs last month", ACCENT1, "O"),
            metric_card("Transactions",        "48,312", "Up 7.4% vs last month",  ACCENT4, "T"),
            metric_card("Anomalies Flagged (Today)", "347",    "+23 in last hour",        ACCENT3, "!"),
            metric_card("Fraud Rate",          "0.72%",  "+0.12% spike today",      ACCENT2, "%"),
        ], style={"display":"flex","gap":"14px","marginBottom":"20px"}),
        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"14px","marginBottom":"14px"}, children=[
            card([dcc.Graph(figure=fig_trend(),     config={"displayModeBar":False})], title="Revenue vs Anomalies (30 days)"),
            card([dcc.Graph(figure=fig_histogram(), config={"displayModeBar":False})], title="Anomaly Score Distribution"),
        ]),
        html.Div(style={"display":"grid","gridTemplateColumns":"1.1fr 1.1fr 0.8fr","gap":"14px"}, children=[
            card([alert_row(a) for a in alerts], title="Live Anomaly Alerts"),
            card([dcc.Graph(figure=fig_shap(),  config={"displayModeBar":False})], title="Feature Importance - SHAP Values"),
            card([
                dcc.Graph(figure=fig_donut(), config={"displayModeBar":False}),
                html.Div([
                    html.Div([
                        html.Div(style={"width":"8px","height":"8px","borderRadius":"50%",
                                        "background":c,"marginRight":"8px","flexShrink":"0"}),
                        html.Div([
                            html.Div(l, style={"fontSize":"9px","color":TEXT_SEC}),
                            html.Div(f"{v:,}", style={"fontSize":"14px","fontWeight":"700","color":TEXT_PRI}),
                        ])
                    ], style={"display":"flex","alignItems":"center","padding":"8px",
                              "background":BG_CARD2,"borderRadius":"6px"})
                    for l,v,c in zip(seg_labels, seg_values, seg_colors)
                ], style={"display":"flex","flexDirection":"column","gap":"6px"})
            ], title="Risk Segmentation"),
        ]),
    ])

def page_fraud_predictor():
    return html.Div([
        section_heading("Fraud Predictor", "Enter transaction details for real-time anomaly risk scoring"),
        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"20px"}, children=[
            card([
                html.Div("Transaction Parameters", style={"fontSize":"13px","fontWeight":"700",
                    "color":TEXT_PRI,"marginBottom":"18px"}),
                html.Div([
                    html.Label("Transaction Amount ($)", style={"fontSize":"11px","color":TEXT_SEC,
                        "textTransform":"uppercase","letterSpacing":"1px",
                        "marginBottom":"6px","display":"block"}),
                    dcc.Input(id="fp-amount", placeholder="e.g. 450", type="number", min=0, style={
                        "width":"100%","background":BG_CARD2,"border":f"1px solid {BORDER}",
                        "borderRadius":"6px","padding":"10px 12px","color":TEXT_PRI,
                        "fontSize":"13px","outline":"none","marginBottom":"14px"}),
                ]),
                html.Div([
                    html.Label("Order Velocity (last 1h)", style={"fontSize":"11px","color":TEXT_SEC,
                        "textTransform":"uppercase","letterSpacing":"1px",
                        "marginBottom":"6px","display":"block"}),
                    dcc.Input(id="fp-velocity", placeholder="e.g. 3", type="number", min=0, style={
                        "width":"100%","background":BG_CARD2,"border":f"1px solid {BORDER}",
                        "borderRadius":"6px","padding":"10px 12px","color":TEXT_PRI,
                        "fontSize":"13px","outline":"none","marginBottom":"14px"}),
                ]),
                html.Div([
                    html.Label("Account Age (days)", style={"fontSize":"11px","color":TEXT_SEC,
                        "textTransform":"uppercase","letterSpacing":"1px",
                        "marginBottom":"6px","display":"block"}),
                    dcc.Input(id="fp-age", placeholder="e.g. 180", type="number", min=0, style={
                        "width":"100%","background":BG_CARD2,"border":f"1px solid {BORDER}",
                        "borderRadius":"6px","padding":"10px 12px","color":TEXT_PRI,
                        "fontSize":"13px","outline":"none","marginBottom":"14px"}),
                ]),
                html.Div([
                    html.Label("IP Risk Score (0-1)", style={"fontSize":"11px","color":TEXT_SEC,
                        "textTransform":"uppercase","letterSpacing":"1px",
                        "marginBottom":"6px","display":"block"}),
                    dcc.Input(id="fp-iprisk", placeholder="e.g. 0.2", type="number", min=0, max=1, step=0.01, style={
                        "width":"100%","background":BG_CARD2,"border":f"1px solid {BORDER}",
                        "borderRadius":"6px","padding":"10px 12px","color":TEXT_PRI,
                        "fontSize":"13px","outline":"none","marginBottom":"14px"}),
                ]),
                html.Div([
                    html.Label("Payment Method", style={"fontSize":"11px","color":TEXT_SEC,
                        "textTransform":"uppercase","letterSpacing":"1px",
                        "marginBottom":"6px","display":"block"}),
                    dcc.Dropdown(
                        id="fp-payment",
                        options=[{"label":m,"value":m} for m in
                                 ["Credit Card","Debit Card","PayPal","UPI","Crypto","COD"]],
                        placeholder="Select payment method",
                        style={"background":BG_CARD2},
                    ),
                ], style={"marginBottom":"14px"}),
                html.Button("Run Anomaly Detection", id="fp-run-btn", n_clicks=0, style={
                    "width":"100%","background":ACCENT1,"color":BG_MAIN,"border":"none",
                    "borderRadius":"6px","padding":"12px","fontSize":"13px",
                    "fontWeight":"700","cursor":"pointer","marginTop":"6px"
                }),
            ]),
            html.Div([
                card([
                    html.Div("Risk Assessment Result", style={"fontSize":"13px","fontWeight":"700",
                        "color":TEXT_PRI,"marginBottom":"18px"}),
                    html.Div(id="fp-result-panel", children=[
                        html.Div([
                            html.Div("—", style={"fontSize":"28px","fontWeight":"800",
                                "color":TEXT_SEC,"letterSpacing":"2px"}),
                            html.Div("Enter values and click Run", style={"fontSize":"13px","color":TEXT_SEC,"marginTop":"6px"}),
                        ], style={"textAlign":"center","padding":"30px","background":BG_CARD2,
                                  "borderRadius":"8px","border":f"1px solid {BORDER}","marginBottom":"16px"}),
                    ]),
                    html.Div(id="fp-confidence-row", children=[]),
                    html.Div(id="fp-detail-rows", children=[]),
                ]),
                html.Div(style={"height":"14px"}),
                card([alert_row(a) for a in alerts[:3]], title="Similar Past Flagged Cases"),
            ]),
        ]),
    ])

def page_dataset():
    return html.Div([
        section_heading("Dataset & EDA",
            "Exploratory data analysis - synthetic e-commerce dataset - 48,312 records"),
        html.Div([
            metric_card("Total Records", "48,312", "100% complete, no nulls", ACCENT1, "D"),
            metric_card("Features",      "14",     "Numeric + categorical",   ACCENT4, "F"),
            metric_card("Fraud Samples", "3,612",  "7.48% class imbalance",   ACCENT3, "!"),
            metric_card("Date Range",    "30 days", DATE_RANGE_LABEL,         ACCENT2, "C"),
        ], style={"display":"flex","gap":"14px","marginBottom":"20px"}),
        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"14px","marginBottom":"14px"}, children=[
            card([dcc.Graph(figure=fig_daily_orders(),    config={"displayModeBar":False})], title="Daily Orders vs Returns"),
            card([dcc.Graph(figure=fig_payment_methods(), config={"displayModeBar":False})], title="Fraud Rate by Payment Method"),
        ]),
        card([
            html.Table(style={"width":"100%","borderCollapse":"collapse","fontSize":"12px"}, children=[
                html.Thead(html.Tr([
                    html.Th(h, style={"textAlign":"left","padding":"10px 14px","borderBottom":f"1px solid {BORDER}",
                        "color":TEXT_SEC,"fontSize":"10px","textTransform":"uppercase","letterSpacing":"1px"})
                    for h in ["Feature","Type","Min","Max","Mean","Std Dev","Missing"]
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td(c, style={"padding":"9px 14px","borderBottom":f"1px solid {BORDER}",
                            "color":TEXT_PRI if i==0 else TEXT_SEC})
                        for i, c in enumerate(row)
                    ])
                    for row in [
                        ["transaction_amount","float","$1","$12,400","$178","$312","0%"],
                        ["order_velocity_1h","int","0","47","2.3","4.1","0%"],
                        ["account_age_days","int","0","3650","412","318","0%"],
                        ["ip_risk_score","float","0.0","1.0","0.18","0.21","0%"],
                        ["device_fingerprint","str","-","-","-","-","0%"],
                        ["billing_ship_match","bool","0","1","0.82","0.38","0%"],
                        ["promo_code_used","bool","0","1","0.31","0.46","0%"],
                        ["payment_method","str","-","-","-","-","0%"],
                        ["category","str","-","-","-","-","0%"],
                        ["anomaly_label","int","0","1","0.075","0.26","0%"],
                    ]
                ])
            ])
        ], title="Dataset Schema and Statistics"),
    ])

def page_statistical():
    return html.Div([
        section_heading("Statistical Analysis","Correlation matrix, distributions, and hypothesis testing"),
        html.Div([
            metric_card("Pearson Corr (txn-fraud)", "0.68", "Strong positive signal",    ACCENT1, "~"),
            metric_card("Chi2 p-value",             "0.003","Statistically significant", ACCENT3, "X"),
            metric_card("Class Imbalance Ratio",    "1:12.4","SMOTE applied",            ACCENT2, "="),
            metric_card("Outlier % (IQR)",          "8.2%", "Above 3 sigma",             ACCENT4, "S"),
        ], style={"display":"flex","gap":"14px","marginBottom":"20px"}),
        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"14px"}, children=[
            card([dcc.Graph(figure=fig_corr_matrix(),  config={"displayModeBar":False})], title="Feature Correlation Matrix"),
            card([dcc.Graph(figure=fig_amount_dist(),  config={"displayModeBar":False})], title="Transaction Amount Distribution - Fraud vs Legitimate"),
        ]),
    ])

def page_visualizations():
    return html.Div([
        section_heading("Data Visualizations","Multi-dimensional views of the fraud dataset"),
        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"14px","marginBottom":"14px"}, children=[
            card([dcc.Graph(figure=fig_bubble(),   config={"displayModeBar":False})], title="Fraud Rate vs Avg Order Value by Category"),
            card([dcc.Graph(figure=fig_segment(),  config={"displayModeBar":False})], title="Customer Risk by Acquisition Channel"),
        ]),
        html.Div(style={"display":"grid","gridTemplateColumns":"0.8fr 1.2fr","gap":"14px"}, children=[
            card([
                dcc.Graph(figure=fig_donut(), config={"displayModeBar":False}),
                html.Div([
                    html.Div([
                        html.Div(style={"width":"10px","height":"10px","borderRadius":"50%",
                                        "background":c,"marginRight":"10px","flexShrink":"0"}),
                        html.Div([
                            html.Div(l, style={"fontSize":"11px","color":TEXT_SEC}),
                            html.Div(f"{v:,}", style={"fontSize":"16px","fontWeight":"700","color":TEXT_PRI}),
                        ])
                    ], style={"display":"flex","alignItems":"center","padding":"10px",
                              "background":BG_CARD2,"borderRadius":"6px"})
                    for l,v,c in zip(seg_labels, seg_values, seg_colors)
                ], style={"display":"flex","flexDirection":"column","gap":"8px"})
            ], title="Risk Segmentation"),
            card([dcc.Graph(figure=fig_histogram(), config={"displayModeBar":False})], title="Anomaly Score Distribution"),
        ]),
    ])

def page_model_benchmarks():
    return html.Div([
        section_heading("Model Benchmarks","6 ML models compared - XGBoost selected for production"),
        html.Div([
            metric_card("Best Accuracy", "94%",    "XGBoost",    ACCENT1, "A"),
            metric_card("Best F1-Score", "0.92",   "XGBoost",    ACCENT4, "F"),
            metric_card("Best Recall",   "92%",    "XGBoost",    ACCENT1, "R"),
            metric_card("Prod Model",    "XGBoost","AUC 0.96",   ACCENT2, "P"),
        ], style={"display":"flex","gap":"14px","marginBottom":"20px"}),
        card([dcc.Graph(figure=fig_model_bench(), config={"displayModeBar":False})],
             title="Model Comparison - Accuracy, Precision, Recall, F1",
             extra_style={"marginBottom":"14px"}),
        
        html.Div(
            "Note: Confusion Matrix reflects XGBoost (supervised, production classifier, AUC 0.96). "
            "ROC Curve reflects Isolation Forest (unsupervised anomaly detector, AUC 0.91). "
            "Both models run in parallel — XGBoost for labeled classification, Isolation Forest for unsupervised anomaly scoring.",
            style={"fontSize":"11px","color":TEXT_SEC,"background":BG_CARD2,
                   "border":f"1px solid {BORDER}","borderLeft":f"3px solid {ACCENT2}",
                   "borderRadius":"6px","padding":"10px 14px","marginBottom":"14px","lineHeight":"1.7"}
        ),
        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"14px"}, children=[
            card([dcc.Graph(figure=fig_confusion(), config={"displayModeBar":False})],
                 title="Confusion Matrix - XGBoost (Production Classifier)"),
            card([dcc.Graph(figure=fig_roc(),       config={"displayModeBar":False})],
                 title="ROC Curve - Isolation Forest (Anomaly Detector, AUC 0.91)"),
        ]),
    ])

def page_system_design():
    return html.Div([
        section_heading("System Design","Architecture of the FraudSense ML pipeline"),
        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr 1fr","gap":"14px","marginBottom":"20px"}, children=[
            card([
                html.Div("Data Ingestion", style={"fontSize":"15px","fontWeight":"700","color":TEXT_PRI,"marginBottom":"8px"}),
                html.Div("Real-time transaction stream via Kafka. Batch historical loads from S3. Schema validated on ingest.",
                         style={"fontSize":"12px","color":TEXT_SEC,"lineHeight":"1.8"}),
            ]),
            card([
                html.Div("Feature Engineering", style={"fontSize":"15px","fontWeight":"700","color":TEXT_PRI,"marginBottom":"8px"}),
                html.Div("Velocity features via Redis sliding windows. Device fingerprinting. One-hot encoding. SMOTE for class balancing.",
                         style={"fontSize":"12px","color":TEXT_SEC,"lineHeight":"1.8"}),
            ]),
            card([
                html.Div("ML Inference", style={"fontSize":"15px","fontWeight":"700","color":TEXT_PRI,"marginBottom":"8px"}),
                html.Div("XGBoost served via FastAPI. Isolation Forest for unsupervised anomalies. Under 50ms P99 latency. MLflow versioning.",
                         style={"fontSize":"12px","color":TEXT_SEC,"lineHeight":"1.8"}),
            ]),
        ]),
        card([
            html.Div(style={
                "display":"flex","alignItems":"center","justifyContent":"space-around",
                "flexWrap":"wrap","gap":"10px","padding":"16px"
            }, children=[
                html.Div([
                    html.Div(name, style={"fontSize":"12px","fontWeight":"700","color":TEXT_PRI,"marginBottom":"4px"}),
                    html.Div(sub,  style={"fontSize":"10px","color":TEXT_SEC}),
                ], style={"textAlign":"center","padding":"16px 20px","background":BG_CARD2,
                          "borderRadius":"8px","border":f"1px solid {BORDER}"})
                for name, sub in [
                    ("E-Commerce API","Source"),("Kafka","Streaming"),("Redis","Feature store"),
                    ("XGBoost","ML model"),("FastAPI","REST API"),("Dash","Dashboard"),
                ]
            ])
        ], title="End-to-End Pipeline", extra_style={"marginBottom":"14px"}),
        card([
            html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"20px"}, children=[
                html.Div([
                    html.Div("Tech Stack", style={"fontSize":"10px","fontWeight":"600","color":TEXT_SEC,
                        "textTransform":"uppercase","letterSpacing":"1.5px","marginBottom":"12px"}),
                    *[html.Div([
                        html.Span(tech,    style={"fontSize":"12px","color":TEXT_PRI}),
                        html.Span(purpose, style={"fontSize":"11px","color":TEXT_SEC}),
                    ], style={"display":"flex","justifyContent":"space-between",
                              "padding":"8px 0","borderBottom":f"1px solid {BORDER}"})
                      for tech, purpose in [
                        ("Python 3.11","Core language"),("XGBoost / Sklearn","ML models"),
                        ("Isolation Forest","Anomaly detection"),("Dash + Plotly","Dashboard"),
                        ("FastAPI","Model serving"),("Redis","Feature store"),
                        ("Kafka","Stream processing"),("MLflow","Experiment tracking"),
                    ]],
                ]),
                html.Div([
                    html.Div("Performance SLAs", style={"fontSize":"10px","fontWeight":"600","color":TEXT_SEC,
                        "textTransform":"uppercase","letterSpacing":"1.5px","marginBottom":"12px"}),
                    *[html.Div([
                        html.Span(metric, style={"fontSize":"12px","color":TEXT_PRI}),
                        html.Span(value,  style={"fontSize":"11px","color":color,"fontWeight":"600"}),
                    ], style={"display":"flex","justifyContent":"space-between",
                              "padding":"8px 0","borderBottom":f"1px solid {BORDER}"})
                      for metric, value, color in [
                        ("Inference P99 latency","<50ms",ACCENT1),
                        ("Throughput","10k TPS",ACCENT4),
                        ("Model accuracy","94%",ACCENT1),
                        ("False positive rate","1.88%",ACCENT2),
                        ("Uptime SLA","99.9%",ACCENT1),
                        ("Data freshness","<2s",ACCENT4),
                    ]],
                ]),
            ])
        ], title="Tech Stack and Performance Targets"),
    ])


def nav_item(label, icon, href="/", active=False):
    return dcc.Link(
        html.Div([
            html.Span(icon,  style={"fontSize":"14px","width":"20px","flexShrink":"0"}),
            html.Span(label, style={"fontSize":"13px"}),
        ], style={
            "display":"flex","alignItems":"center","gap":"10px","padding":"9px 14px",
            "borderRadius":"6px","cursor":"pointer",
            "color":  ACCENT1 if active else TEXT_SEC,
            "background": "rgba(0,229,160,0.08)" if active else "transparent",
            "marginBottom":"2px",
        }),
        href=href, style={"textDecoration":"none"}
    )


FONT_CSS = """
* { box-sizing: border-box; }
body { margin: 0; background: #0d1117; font-family: Arial, sans-serif; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #222d3a; border-radius: 2px; }
a { text-decoration: none; }
"""

app = dash.Dash(__name__, title="FraudSense - Anomaly Detection",
                suppress_callback_exceptions=True)

app.index_string = f"""<!DOCTYPE html>
<html>
<head>
    {{%metas%}}
    <title>{{%title%}}</title>
    {{%favicon%}}
    {{%css%}}
    <style>{FONT_CSS}</style>
</head>
<body>
    {{%app_entry%}}
    <footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
</body>
</html>"""


NAV_GROUPS = {
    "MAIN": [
        ("/",               "Overview",             "O"),
        ("/fraud-predictor","Fraud Predictor",       "P"),
    ],
    "ANALYSIS": [
        ("/dataset",        "Dataset & EDA",         "D"),
        ("/statistical",    "Statistical Analysis",  "S"),
        ("/visualizations", "Visualizations",        "V"),
    ],
    "ML": [
        ("/model-benchmarks","Model Benchmarks",     "M"),
        ("/system-design",   "System Design",        "A"),
    ],
}

PAGE_TITLES = {
    "/":                 ("System Integrity Dashboard",   "Real-time fraud detection - Isolation Forest model"),
    "/fraud-predictor":  ("Fraud Predictor",              "Real-time anomaly scoring for individual transactions"),
    "/dataset":          ("Dataset & EDA",                "48,312 synthetic e-commerce transaction records"),
    "/statistical":      ("Statistical Analysis",         "Correlations, distributions, and hypothesis tests"),
    "/visualizations":   ("Data Visualizations",          "Multi-dimensional views of the fraud dataset"),
    "/model-benchmarks": ("Model Benchmarks",             "6 ML models compared - XGBoost selected for production"),
    "/system-design":    ("System Design",                "End-to-end ML pipeline architecture and tech stack"),
}

app.layout = html.Div(
    style={"display":"flex","minHeight":"100vh","background":BG_MAIN},
    children=[
        dcc.Location(id="url", refresh=False),

        # Sidebar (fixed position)
        html.Div(id="sidebar", style={
            "width":"215px","minWidth":"215px","background":BG_SIDEBAR,
            "borderRight":f"1px solid {BORDER}","display":"flex",
            "flexDirection":"column","position":"fixed","top":"0","left":"0",
            "height":"100vh","overflowY":"auto","zIndex":"100"
        }),

        # Main area pushed right by sidebar width
        html.Div(style={"flex":"1","overflowY":"auto","marginLeft":"215px"}, children=[
            html.Div(id="topbar"),
            html.Div(id="page-content", style={"padding":"24px 28px"}),
        ]),
    ]
)


@app.callback(Output("sidebar", "children"), Input("url", "pathname"))
def render_sidebar(pathname):
    if not pathname:
        pathname = "/"
    return [
        # Logo
        html.Div([
            html.Div("FS", style={
                "width":"34px","height":"34px","borderRadius":"8px","background":ACCENT1,
                "display":"flex","alignItems":"center","justifyContent":"center",
                "fontSize":"14px","fontWeight":"800","color":BG_MAIN,"marginBottom":"4px"
            }),
            html.Div("FraudSense", style={"fontSize":"14px","fontWeight":"700","color":TEXT_PRI}),
            html.Div("ANOMALY DETECTION", style={"fontSize":"9px","color":ACCENT1,
                "letterSpacing":"1.5px","textTransform":"uppercase"}),
        ], style={"padding":"22px 16px 18px","borderBottom":f"1px solid {BORDER}"}),

        # Nav links
        html.Div(style={"padding":"0 8px","flex":"1"}, children=[
            item
            for group, links in NAV_GROUPS.items()
            for item in [
                html.Div(group, style={"fontSize":"9px","color":TEXT_SEC,"letterSpacing":"1.5px",
                                       "padding":"14px 14px 6px","textTransform":"uppercase"}),
                *[nav_item(label, icon, href, active=(pathname == href))
                  for href, label, icon in links]
            ]
        ]),

        # Status footer
        html.Div([
            html.Div(style={"width":"8px","height":"8px","borderRadius":"50%",
                            "background":ACCENT1,"marginRight":"8px","flexShrink":"0"}),
            html.Div([
                html.Div("Models Online", style={"fontSize":"11px","fontWeight":"600","color":ACCENT1}),
                html.Div("XGBoost (Prod) · IF Anomaly", style={"fontSize":"10px","color":TEXT_SEC}),
            ])
        ], style={"display":"flex","alignItems":"center","padding":"14px 16px",
                  "borderTop":f"1px solid {BORDER}","background":"rgba(0,229,160,0.05)"}),
    ]


@app.callback(Output("topbar", "children"), Input("url", "pathname"))
def render_topbar(pathname):
    if not pathname:
        pathname = "/"
    title, subtitle = PAGE_TITLES.get(pathname, ("FraudSense", ""))
    return html.Div(style={
        "borderBottom":f"1px solid {BORDER}","padding":"16px 28px",
        "display":"flex","justifyContent":"space-between","alignItems":"center",
        "position":"sticky","top":"0","background":BG_MAIN,"zIndex":"10"
    }, children=[
        html.Div([
            html.Div(title,    style={"fontSize":"22px","fontWeight":"800","color":TEXT_PRI}),
            html.Div(subtitle, style={"fontSize":"11px","color":TEXT_SEC,"marginTop":"2px"}),
        ]),
        html.Div([
            html.Div("Q2 2025", style={"fontSize":"10px","fontWeight":"600","padding":"4px 12px",
                "borderRadius":"4px","background":"rgba(78,158,255,0.12)","color":ACCENT4,"marginRight":"8px"}),
            html.Div("Live Data", style={"fontSize":"10px","fontWeight":"600","padding":"4px 12px",
                "borderRadius":"4px","background":"rgba(0,229,160,0.12)","color":ACCENT1}),
        ], style={"display":"flex","alignItems":"center"}),
    ])


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page(pathname):
    if not pathname or pathname == "/":
        return page_overview()
    elif pathname == "/fraud-predictor":
        return page_fraud_predictor()
    elif pathname == "/dataset":
        return page_dataset()
    elif pathname == "/statistical":
        return page_statistical()
    elif pathname == "/visualizations":
        return page_visualizations()
    elif pathname == "/model-benchmarks":
        return page_model_benchmarks()
    elif pathname == "/system-design":
        return page_system_design()
    else:
        return html.Div([
            html.Div("404", style={"fontSize":"72px","fontWeight":"800","color":ACCENT3}),
            html.Div("Page not found", style={"fontSize":"18px","color":TEXT_SEC,"marginTop":"8px"}),
        ], style={"textAlign":"center","padding":"80px"})



@app.callback(
    Output("fp-result-panel",    "children"),
    Output("fp-confidence-row",  "children"),
    Output("fp-detail-rows",     "children"),
    Input("fp-run-btn", "n_clicks"),
    State("fp-amount",  "value"),
    State("fp-velocity","value"),
    State("fp-age",     "value"),
    State("fp-iprisk",  "value"),
    State("fp-payment", "value"),
    prevent_initial_call=True,
)
def run_fraud_predictor(n_clicks, amount, velocity, age, ip_risk, payment):
    
    amount   = float(amount   or 100)
    velocity = float(velocity or 1)
    age      = float(age      or 365)
    ip_risk  = float(ip_risk  or 0.1)
    pay_enc  = float(_PAYMENT_ENC.get(payment, 0))

    
    X = np.array([[amount, velocity, age, ip_risk, pay_enc]])
    raw_score  = -_IF_MODEL.score_samples(X)[0]          # higher = more anomalous
    prediction = _IF_MODEL.predict(X)[0]                 # -1 = anomaly, 1 = normal
    
    anomaly_score = round(float(np.clip((raw_score - 0.30) / 0.40, 0.0, 1.0)), 2)
    confidence    = round((1 - abs(anomaly_score - 0.5)) * 100 + 40, 1)
    confidence    = min(confidence, 99.0)

    
    if prediction == -1 or anomaly_score >= 0.65:
        risk_label  = "HIGH RISK"
        risk_color  = ACCENT3
        border_col  = ACCENT3
        txn_risk    = ("Transaction risk",  "High",    ACCENT3)
        vel_check   = ("Velocity check",    "Elevated" if velocity > 5 else "Flagged", ACCENT3)
        ip_rep      = ("IP reputation",     "Risky"    if ip_risk > 0.5 else "Suspicious", ACCENT3)
        dev_trust   = ("Device trust",      "Low",     ACCENT3)
    elif anomaly_score >= 0.40:
        risk_label  = "MEDIUM RISK"
        risk_color  = ACCENT2
        border_col  = ACCENT2
        txn_risk    = ("Transaction risk",  "Moderate", ACCENT2)
        vel_check   = ("Velocity check",    "Elevated" if velocity > 3 else "Normal", ACCENT2)
        ip_rep      = ("IP reputation",     "Moderate" if ip_risk > 0.3 else "Clean", ACCENT2)
        dev_trust   = ("Device trust",      "Medium",  ACCENT2)
    else:
        risk_label  = "LOW RISK"
        risk_color  = ACCENT1
        border_col  = ACCENT1
        txn_risk    = ("Transaction risk",  "Low",    ACCENT1)
        vel_check   = ("Velocity check",    "Normal", ACCENT1)
        ip_rep      = ("IP reputation",     "Clean",  ACCENT1)
        dev_trust   = ("Device trust",      "High",   ACCENT1)

    
    result_panel = html.Div([
        html.Div(risk_label, style={"fontSize":"28px","fontWeight":"800",
            "color":risk_color,"letterSpacing":"2px"}),
        html.Div(f"Anomaly Score: {anomaly_score:.2f}",
            style={"fontSize":"14px","color":TEXT_SEC,"marginTop":"6px"}),
    ], style={"textAlign":"center","padding":"30px","background":BG_CARD2,
              "borderRadius":"8px","border":f"1px solid {border_col}44","marginBottom":"16px"})

    confidence_pct = f"{confidence:.1f}%"
    confidence_row = [
        html.Div(f"Model Confidence: {confidence_pct}",
            style={"fontSize":"11px","color":TEXT_SEC,
                   "textTransform":"uppercase","letterSpacing":"1px","marginBottom":"8px"}),
        html.Div(style={"height":"8px","background":BORDER,"borderRadius":"4px","marginBottom":"18px"}, children=[
            html.Div(style={"height":"100%","width":confidence_pct,"background":risk_color,"borderRadius":"4px"})
        ]),
    ]

    detail_rows = [
        html.Div([
            html.Span(label, style={"fontSize":"12px","color":TEXT_SEC}),
            html.Span(value, style={"fontSize":"12px","color":color,"fontWeight":"600"}),
        ], style={"display":"flex","justifyContent":"space-between","padding":"8px 0",
                  "borderBottom":f"1px solid {BORDER}"})
        for label, value, color in [txn_risk, dev_trust, vel_check, ip_rep]
    ]

    return result_panel, confidence_row, detail_rows



if __name__ == "__main__":
    print("\n" + "="*50)
    print("  FraudSense Dashboard - Navigation Edition")
    print("  Open: http://127.0.0.1:8050")
    print("="*50)
    print("\n  Pages available:")
    for href, (title, _) in PAGE_TITLES.items():
        print(f"  {href:<22} -> {title}")
    print()
    app.run(debug=True, host="127.0.0.1", port=8050)
