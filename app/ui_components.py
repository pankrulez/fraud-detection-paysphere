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
        
        # NATIVE PLOTLY RENDER: This guarantees full interactivity, hover templates, and responsiveness
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def info_card(title, content_html, accent="primary"):
    """
    A text-based card for metrics, pipeline steps, or general information.
    Replaces your previous section_card and heading_card.
    """
    color = ACCENT_COLORS.get(accent, accent)
    
    with st.container(border=True):
        st.markdown(f"""
            <div style="border-left: 4px solid {color}; padding-left: 12px; margin-bottom: 12px;">
                <h3 style="color: {color}; margin: 0; font-weight: 600;">{title}</h3>
            </div>
            <div style="padding-left: 8px; color: #e5e7eb;">
                {content_html}
            </div>
        """, unsafe_allow_html=True)