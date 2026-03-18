import streamlit as st

# ==============================
# DESIGN SYSTEM
# ==============================
ACCENT_COLORS = {
    "primary": "#4C8BF5",      # Blue
    "success": "#5CA27F",      # Muted green
    "warning": "#B08968",      # Bronze
    "info": "#7E8CE0",         # Indigo
    "neutral": "#5DA9A6",      # Teal-grey
    "governance": "#9A7AA0",   # Muted violet
}

DEFAULT_CHART_HEIGHT = 420

st.markdown("""
    <style>
    /* Glassmorphism effect for the sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(10px);
    }
    /* Pulse animation for the Online status */
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    .pulse-dot {
        animation: pulse 2s infinite;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================
# NATIVE STREAMLIT CARDS
# ==============================

def chart_card(title, description, fig, accent="primary", height=DEFAULT_CHART_HEIGHT):
    """
    Renders a native Streamlit container with a custom styled title, 
    description, and a fully interactive Plotly chart.
    """
    color = ACCENT_COLORS.get(accent, accent)
    
    # Use native Streamlit container for the card background and border
    with st.container(border=True):
        
        # Inject custom HTML *only* for the text/headers to keep your beautiful left-border accent
        st.markdown(f"""
            <div style="border-left: 4px solid {color}; padding-left: 12px; margin-bottom: 16px;">
                <h3 style="color: {color}; margin: 0; font-weight: 600;">{title}</h3>
                <p style="margin: 4px 0 0 0; color: #a1a1aa; font-size: 0.95rem;">{description}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Update figure margins so it fits snugly in the card
        fig.update_layout(
            height=height, 
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", # Transparent background to inherit the container's dark mode
            plot_bgcolor="rgba(0,0,0,0)"
        )
        
        # NATIVE PLOTLY RENDER
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def info_card(title, content_html, accent="primary"):
    """
    A text-based card for metrics, pipeline steps, or general information.
    Replaces your previous section_card and heading_card.
    """
    color = ACCENT_COLORS.get(accent, accent)
    
    st.markdown(f"""
        <div style="
            border: 1px solid #334155; 
            border-left: 4px solid {color}; 
            padding: 20px; 
            border-radius: 12px; 
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
            margin-bottom: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
            <h3 style="color: {color}; margin: 0 0 12px 0; font-weight: 600; font-size: 1.1rem;">{title}</h3>
            <div style="color: #e5e7eb; line-height: 1.6; font-size: 0.9rem;">
                {content_html}
            </div>
        </div>
    """, unsafe_allow_html=True)
        

def render_threshold_explanation(threshold):
    """
    Translates technical thresholds into business strategy with explicit bounds.
    """
    if threshold < 0.03:
        posture = "🛡️ Aggressive Protection"
        bounds = "Threshold < 0.030"
        description = "Prioritizing maximum fraud capture. Ideal for high-risk periods or identifying subtle botnet patterns."
    elif threshold > 0.149:
        posture = "🚀 Growth Focused"
        description = "Prioritizing seamless customer journeys. Ideal for low-risk scenarios or major sales events to maximize conversion."
        bounds = "Threshold > 0.149"
    else:
        posture = "⚖️ Balanced Risk"
        description = "The 'Golden Mean'. Optimizing the trade-off between stopping thieves and serving honest customers."
        bounds = "0.030 ≤ Threshold ≤ 0.149"

    st.markdown(f"""
        <div style="background: rgba(59, 130, 246, 0.1); padding: 20px; border-radius: 12px; 
                    border-left: 5px solid #3B82F6; margin-bottom: 25px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="margin:0; color: #60A5FA;">Current Strategy: {posture}</h4>
                <span style="background: #1E293B; color: #3B82F6; padding: 4px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; border: 1px solid #3B82F6;">
                    {bounds}
                </span>
            </div>
            <p style="margin:10px 0 8px 0; color: #CBD5E1; font-size: 0.95rem; line-height: 1.4;">{description}</p>
            <div style="border-top: 1px solid rgba(255,255,255,0.1); padding-top: 8px;">
                <span style="color: #94A3B8; font-size: 0.85rem;">Active Sensitivity: </span>
                <span style="color: #F8FAFC; font-weight: 600; font-size: 0.85rem;">{threshold:.3f}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)