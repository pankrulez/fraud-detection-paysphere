import streamlit as st
import pandas as pd
import time
from datetime import datetime

def render_batch_processing(scorer):
    st.title("📂 Bulk Risk Assessment")
    st.markdown("""
        Upload a transaction manifest (CSV) to perform high-speed batch scoring. 
        This module uses **Vectorized Inference** and **Chunked API Requests** to handle large datasets efficiently.
    """)

    uploaded_file = st.file_uploader("Upload Transaction Manifest", type="csv")

    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        st.info(f"📋 Dataset Loaded: **{len(df_upload):,}** transactions detected.")

        # --- DATA CONTRACT VALIDATION ---
        required_cols = ['amount', 'payment_method', 'merchant_category', 'ip_address_risk_score', 'device_trust_score']
        missing = [col for col in required_cols if col not in df_upload.columns]
        
        if missing:
            st.error(f"❌ **Schema Validation Failed.** Missing required columns: {missing}")
            return

        if st.button("🚀 Execute Batch Scoring", type="primary", use_container_width=True):
            # Define chunking parameters
            CHUNK_SIZE = 5000
            all_probs = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            start_time = time.time()

            try:
                # Progressively process the dataframe in chunks
                for i in range(0, len(df_upload), CHUNK_SIZE):
                    chunk = df_upload.iloc[i : i + CHUNK_SIZE].copy()
                    
                    # Ensure schema alignment for this chunk
                    chunk = align_batch_schema(chunk)
                    
                    current_chunk_num = (i // CHUNK_SIZE) + 1
                    total_chunks = (len(df_upload) // CHUNK_SIZE) + 1
                    status_text.text(f"Processing Chunk {current_chunk_num}/{total_chunks}...")
                    
                    # API CALL
                    probs = scorer.predict_proba_batch(chunk)
                    all_probs.extend(probs)
                    
                    # Update Progress
                    progress_bar.progress(min(1.0, (i + CHUNK_SIZE) / len(df_upload)))

                duration = time.time() - start_time
                st.success(f"✅ Batch Scoring Complete in {duration:.2f} seconds!")

                # Attach results
                df_upload['fraud_probability'] = all_probs
                df_upload['is_flagged'] = (df_upload['fraud_probability'] >= scorer.threshold).astype(int)

                # Display Summary Metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("Flagged Transactions", f"{df_upload['is_flagged'].sum():,}")
                c2.metric("Total Risk Exposure", f"₹{df_upload[df_upload['is_flagged']==1]['amount'].sum():,.0f}")
                c3.metric("Avg Risk Score", f"{df_upload['fraud_probability'].mean():.2%}")

                st.divider()
                st.subheader("Results Preview")
                st.dataframe(df_upload.head(100), use_container_width=True)

                # Download Result
                csv = df_upload.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Scored Manifest",
                    data=csv,
                    file_name=f"scored_manifest_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"Batch Processing Error: {str(e)}")

def align_batch_schema(df):
    """Fills missing fields with defaults to satisfy the API Pydantic contract."""
    defaults = {
        "customer_id": "C_BATCH", "device_id": "D_BATCH", "merchant_id": "M_BATCH",
        "timestamp": datetime.utcnow().isoformat(), "is_international": 0,
        "is_weekend": 0, "txn_count_last_24h": 1, "hour_of_day": 12, "day_of_week": 0,
        "merchant_historical_fraud_rate": 0.05
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
    return df