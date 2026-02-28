import streamlit as st

def section_card(title, content_html, accent="#4C8BF5"):
    html = f"""
<div style="
    background-color:#151b26;
    padding:24px;
    border-radius:12px;
    border-left:4px solid {accent};
    border-top:1px solid rgba(255,255,255,0.06);
    border-right:1px solid rgba(255,255,255,0.06);
    border-bottom:1px solid rgba(255,255,255,0.06);
    margin-bottom:28px;
">
    <h2 style="color:{accent}; margin-bottom:14px; font-weight:600;">
        {title}
    </h2>

    {content_html}

</div>
"""
    st.html(html)
    
# chart card    
def chart_card(title, description, fig, accent="#4C8BF5", height=400):
    plot_html = fig.to_html(
        full_html=False,
        include_plotlyjs=True,  # Bundle Plotly.js inside (fixes CDN issues)
        config={"displayModeBar": False, "responsive": True},
    )

    html = f"""
<div style="
    background-color:#151b26;
    padding:24px;
    border-radius:12px;
    border-left:4px solid {accent};
    border-top:1px solid rgba(255,255,255,0.06);
    border-right:1px solid rgba(255,255,255,0.06);
    border-bottom:1px solid rgba(255,255,255,0.06);
    margin-bottom:28px;
">
    <h3 style="color:{accent}; margin-bottom:8px; font-weight:600;">
        {title}
    </h3>
    <p style="margin-bottom:16px; color:#ccc;">
        {description}
    </p>
    <div style="height:{height}px; width:100%;">
        {plot_html}
    </div>
</div>
"""
    st.html(html)


def heading_card(heading, accent="#4C8BF5"):
    html = f"""
<div style="
    background-color:#151b26;
    padding:5px;
    border-radius:12px;
    border-left:4px solid {accent};
    border-top:1px solid rgba(255,255,255,0.06);
    border-right:1px solid rgba(255,255,255,0.06);
    border-bottom:1px solid rgba(255,255,255,0.06);
    margin-bottom:10px;
">
    <h2 style="color:{accent}; margin-bottom:14px; font-weight:600;">
        {heading}
    </h2>

</div>
"""
    st.html(html)