import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title
import style

def pages_users():
    show_pages(
        [
            Page("main.py", "Home", "🏠"),
            # # Page("Not in Section",in_section=True),
            Page("data.py", "Data Exploration", icon="📄", in_section=False),
            Page("feature.py", "Feature Extraction", icon='📄', in_section=False),
            Page("model_train.py", "Model Details", icon='📄', in_section=False),
            Page("patient_readmission.py", "Patient Re-addmission", icon='📄', in_section=False),

        ]
    )
if __name__ == "__main__":
    style.Style(title='Home', header=f"")
    st.markdown(
        f'''<p class="small-font" style="font-size: 30px; color:orange;font-weight:450;font-style: italic;">
                                                <strong>Welcome to the DiabeVista!!</strong></p>
                                                ''',
        unsafe_allow_html=True)
    st.markdown(
        f'''<p class="small-font" style="font-size: 18px;font-weight:450;text-align:justify;">
            <strong style="color:orange;font-size:18px;">Overview:</strong><br>
            &emsp;&emsp;&emsp;This dashboard provides a comprehensive analysis of diabetes patient readmission using machine learning and data visualization techniques. Our goal is to understand key factors influencing readmission rates and to develop effective predictive models for better patient management.<br><br>
            <strong style="color:orange;font-size:18px;text-align:justify;">Key Insights:</strong>
            <ul style ="font-size:18px;" >
                &emsp;&emsp;<strong>Model Accuracy:</strong><br> &emsp;&emsp;&emsp;&emsp;Achieved up to 96.15% accuracy with KNN, Decision Tree, and Random Forest models.<br>
                &emsp;&emsp;<strong>Critical Factors:</strong><br>&emsp;&emsp;&emsp;&emsp; Identified key features influencing readmission rates.<br>
                &emsp;&emsp;<strong>Next Steps:</strong> <br>&emsp;&emsp;&emsp;&emsp;Enhancing model performance and collaborating with healthcare experts.<br>
            </ul>
            <span style="font-size:18px;">Explore our dashboard for actionable insights into diabetes management.</span></p>
            ''',
        unsafe_allow_html=True
    )
    pages_users()

