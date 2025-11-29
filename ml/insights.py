def generate_insights(df, results):
    insights = []

    insights.append(f"Total rows: {len(df)}")
    insights.append(f"Total columns: {df.shape[1]}")

    # Top correlated features
    corr = df.corr(numeric_only=True).abs().unstack().sort_values(ascending=False)
    top_corr = corr[corr != 1].head(3)

    insights.append("\nTop Correlated Feature Pairs:")
    for pair, value in top_corr.items():
        insights.append(f"{pair[0]} ↔ {pair[1]}: {value:.2f}")
    
    # Best model
    best_model = results.sort_values(by="Accuracy", ascending=False).iloc[0]
    insights.append(f"\nBest Model: {best_model['Model']} with accuracy {best_model['Accuracy']:.2f}")

    return "\n".join(insights)

