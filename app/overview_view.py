import streamlit as st
import streamlit.components.v1 as components


def animated_counter(label, value, color, suffix=""):
    components.html(f"""
        <div style="text-align:center; padding:20px;">
            <h2 id="{label}" style="margin:0; font-size:2.2rem; color:{color};">0</h2>
            <p style="margin:0; color:#94a3b8;">{label}</p>
        </div>

        <script>
        let count = 0;
        let target = {value};
        let speed = Math.ceil(target / 50);

        let interval = setInterval(() => {{
            count += speed;
            if(count >= target) {{
                count = target;
                clearInterval(interval);
            }}
            document.getElementById("{label}").innerText =
                count.toLocaleString() + "{suffix}";
        }}, 20);
        </script>
    """, height=120)


def render_overview(load_sample_data_fn):

    st.markdown("### 🏠 Project Overview")

    df = load_sample_data_fn()
    total_txn = len(df)
    fraud_count = int(df["is_fraud"].sum())
    fraud_rate = round(df["is_fraud"].mean() * 100, 2)

    # HERO SECTION
    st.markdown("""
    <div class="glass-card">
        <h3 style="margin-top:0; color:#3b82f6;">Why this project matters</h3>
        <p style="color:#cbd5e1;">
            PaySphere processes millions of digital payments each month. 
            Even a small <span style="color:#ef4444; font-weight:600;">fraud rate</span> 
            translates into significant financial loss and customer churn.
        </p>
        <p style="color:#cbd5e1;">
            This system converts ML predictions into 
            <span style="color:#22c55e; font-weight:600;">clear business actions</span>
            (Allow / Soft Review / OTP / Hard Block) to balance 
            <span style="color:#22c55e;">fraud reduction</span> 
            with <span style="color:#f59e0b;">customer friction</span>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # KPI STRIP (Semantic Colors)
    st.markdown("### 📈 Key Sample Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        animated_counter("Transactions", total_txn, "#3b82f6")

    with col2:
        animated_counter("Fraud Cases", fraud_count, "#ef4444")

    with col3:
        animated_counter("Fraud Rate", fraud_rate, "#ef4444", "%")

    st.markdown("")

    # DEMO FLOW
    st.markdown("""
    <div class="glass-card">
        <h4 style="color:#3b82f6;">Suggested Demo Flow</h4>
        <ol style="color:#cbd5e1;">
            <li><span style="color:#22c55e;">Live Scoring</span> – simulate high & low risk.</li>
            <li><span style="color:#9333ea;">Analytics</span> – visualize patterns & signals.</li>
            <li><span style="color:#3b82f6;">Pipeline</span> – explain architecture end-to-end.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # BUSINESS & TECH HIGHLIGHTS
    st.markdown("### Business & Technical Highlights")

    col_t1, col_t2, col_t3 = st.columns(3)

    with col_t1:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#22c55e;">Business Outcomes</h4>
            <ul style="color:#cbd5e1;">
                <li><span style="color:#22c55e;">Reduce chargebacks</span> and fraud loss.</li>
                <li><span style="color:#22c55e;">Lower false positives</span>.</li>
                <li><span style="color:#3b82f6;">Explainable risk signals</span>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_t2:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#9333ea;">ML Design</h4>
            <ul style="color:#cbd5e1;">
                <li>Tree-based model on rich features.</li>
                <li><span style="color:#ef4444;">SMOTE</span> for imbalance.</li>
                <li><span style="color:#f59e0b;">Threshold tuning</span> for cost trade-offs.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_t3:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#3b82f6;">Engineering & MLOps</h4>
            <ul style="color:#cbd5e1;">
                <li>Modular repo structure.</li>
                <li>Versioned model artifacts.</li>
                <li>CI-ready testing pipeline.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("")

    st.markdown(
        """
        <div style="color:#cbd5e1;">
            <ul>
                <li><span style="color:#22c55e;">Green</span> = positive business impact. </li>
                <li><span style="color:#ef4444;">Red</span> = fraud/risk. </li>
                <li><span style="color:#f59e0b;">Amber</span> = trade-offs. </li>
                <li><span style="color:#3b82f6;">Blue</span> = system layer. </li>
                <li><span style="color:#9333ea;">Purple</span> = ML intelligence. </li>
            </ul>
        </div>
        """, unsafe_allow_html=True
    )