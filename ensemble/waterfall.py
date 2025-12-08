import plotly.graph_objects as go

def plot_confidence_waterfall(result):
    """
    result = { "svm": float, "mlp": float, "resnet": float, "final_prob": float }
    """
    svm = result["svm"]
    mlp = result["mlp"]
    res = result["resnet"]
    final = result["final_prob"]

    # Contributions (rescaled around 0 for waterfall)
    values = [svm, mlp - svm, res - mlp, final - res]

    labels = [
        "SVM Probability",
        "MLP Adjustment",
        "ResNet Adjustment",
        "Final Ensemble"
    ]

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "relative"],
        x=labels,
        text=[f"{v:.3f}" for v in values],
        y=values,
        connector={"line": {"color": "gray"}},
        increasing={"marker":{"color":"#2ecc71"}},
        decreasing={"marker":{"color":"#e74c3c"}},
        totals={"marker":{"color":"#8e44ad"}}
    ))

    fig.update_layout(
        title="Model Confidence Waterfall Chart",
        waterfallgap=0.4,
        yaxis_title="Probability Contribution",
        height=480
    )
    return fig

