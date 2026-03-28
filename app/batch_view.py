import streamlit as st
import pandas as pd
import time
from datetime import datetime
from app.ui_components import info_card

def render_batch_processing(scorer):
    is_online = scorer.check_api_health() if scorer else False
    status_color = "#10B981" if is_online else "#EF4444"
    status_text = "UPLOADER READY" if is_online else "SYSTEM OFFLINE"
    sub_text = "Chunked API Mode" if is_online else "Cannot Process Batch"
    
    # 1. HEADER & STATUS
    col_t, col_s = st.columns([3, 1])
    with col_t:
        st.title("📂 Bulk Risk Assessment")
        st.caption("High-capacity vectorized inference for transaction manifests and historical auditing.")
    
    with col_s:
        st.markdown(f"""
            <div style="background: {status_color}11; border: 1px solid {status_color}; 
                        padding: 12px; border-radius: 10px; text-align: center;">
                <span style="color: {status_color}; font-weight: 700; font-size: 0.85rem;">● {status_text}</span><br>
                <span style="color: #94a3b8; font-size: 0.7rem;">{sub_text}</span>
            </div>
        """, unsafe_allow_html=True)

    st.write("---")

    # 2. UPLOAD SECTION
    uploaded_file = st.file_uploader("Upload Transaction Manifest (CSV)", type="csv", help="Ensure CSV contains 'amount' and 'merchant_category' for best results.")

    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        
        # Immediate Stats
        c1, c2, c3 = st.columns(3)
        with c1:
            info_card("Payload Scale", f"**{len(df_upload):,}** Transactions", accent="primary")
        with c2:
            info_card("Data Density", f"**{len(df_upload.columns)}** Data Points/Row", accent="success")
        with c3:
            # Check for critical missing columns
            required_cols = ['amount', 'payment_method', 'merchant_category']
            missing = [col for col in required_cols if col not in df_upload.columns]
            status = "✅ Valid Schema" if not missing else f"⚠️ Missing: {len(missing)} cols"
            info_card("Schema Integrity", status, accent="warning" if missing else "success")

        if missing:
            st.warning(f"Note: Missing columns {missing} will be auto-filled with system defaults for inference.")

        st.markdown("<br>", unsafe_allow_html=True)

        # 3. EXECUTION ENGINE
        if st.button("🚀 INITIATE BATCH SCORING", type="primary", use_container_width=True):
            CHUNK_SIZE = 5000
            all_probs = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            start_time = time.time()

            try:
                for i in range(0, len(df_upload), CHUNK_SIZE):
                    chunk = df_upload.iloc[i : i + CHUNK_SIZE].copy()
                    chunk = align_batch_schema(chunk)
                    
                    current_chunk_num = (i // CHUNK_SIZE) + 1
                    total_chunks = (len(df_upload) // CHUNK_SIZE) + 1
                    status_text.markdown(f"`Processing Vector Chunk {current_chunk_num}/{total_chunks}...`")
                    
                    # API CALL
                    probs = scorer.predict_proba_batch(chunk)
                    all_probs.extend(probs)
                    
                    progress_bar.progress(min(1.0, (i + CHUNK_SIZE) / len(df_upload)))

                duration = time.time() - start_time
                status_text.empty()
                st.success(f"✅ Batch Scoring Complete | Runtime: {duration:.2f}s")

                # Results Attachment
                df_upload['fraud_probability'] = all_probs
                df_upload['is_flagged'] = (df_upload['fraud_probability'] >= scorer.threshold).astype(int)

                # 4. RESULTS KPI STRIP
                st.write("---")
                r1, r2, r3 = st.columns(3)
                r1.metric("Flagged for Review", f"{df_upload['is_flagged'].sum():,}")
                r2.metric("Total Risk Exposure", f"₹{df_upload[df_upload['is_flagged']==1]['amount'].sum():,.0f}")
                r3.metric("Avg Population Risk", f"{df_upload['fraud_probability'].mean():.2%}")

                st.divider()
                
                # 5. DATA EXFILTRATION
                col_d1, col_d2 = st.columns([2, 1])
                with col_d1:
                    st.subheader("Scored Manifest Preview")
                    st.dataframe(df_upload.head(100), use_container_width=True)
                
                with col_d2:
                    st.subheader("Export Results")
                    csv = df_upload.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Scored CSV",
                        data=csv,
                        file_name=f"paysphere_scored_{datetime.now().strftime('%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    st.info("The exported file includes 'fraud_probability' and 'is_flagged' based on your current active threshold.")

            except Exception as e:
                st.error(f"Batch Processing Error: {str(e)}")

def align_batch_schema(df):
    """Ensures Pydantic compliance by injecting required fields and forcing types."""
    defaults = {
        "customer_id": "C_BATCH", "device_id": "D_BATCH", "merchant_id": "M_BATCH",
        "timestamp": datetime.utcnow().isoformat(), "amount": 100.0, 
        "payment_method": "upi", "merchant_category": "retail",
        "ip_address_risk_score": 0.0, "device_trust_score": 1.0,
        "is_international": 0, "is_weekend": 0, "past_fraud_count_customer": 0, 
        "past_disputes_customer": 0, "txn_count_last_24h": 1, 
        "customer_tenure_days": 365, "location_change_flag": 0, 
        "otp_success_rate_customer": 1.0, "ip_address_country_match": 1, 
        "hour_of_day": 12, "day_of_week": 0, "merchant_historical_fraud_rate": 0.0
    }
    
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
            
    df['amount'] = df['amount'].astype(float)
    df['ip_address_risk_score'] = df['ip_address_risk_score'].astype(float)
    df['device_trust_score'] = df['device_trust_score'].astype(float)
    df['merchant_historical_fraud_rate'] = df['merchant_historical_fraud_rate'].astype(float)
    df['day_of_week'] = df['day_of_week'].astype(int)
    
    return df