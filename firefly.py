import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Hybrid Mental Health Predictor", layout="wide")

# Define consistent class ordering
CLASSES = ["Needs Support (Clinical)", "At risk (Subclinical)", "Healthy (Asymptomatic)"]

# Positive responses mapping (updated to match Google Form options)
POSITIVE_RESPONSES = {
    # Q1 - Daily feelings
    "Happy", "Motivated", "Energetic", "Calm",
    
    # Q2 - Enjoy in education
    "Learning new things", "Meeting new people", "Exploring career options", 
    "Achieving good grades", "Campus environment", "Extracurricular activities",
    
    # Q3 - Desired changes
    "More practical work", "Better time management", "More support from teachers",
    "More interactive classes", "Better study materials",
    
    # Q4 - Solo vs group work
    "Group — I enjoy collaboration", "Group — It helps me learn faster", 
    "Alone — I focus better", "Depends on the task",
    
    # Q5 - Handling stress
    "Study in advance", "Talk to friends/family", "Meditate or exercise",
    "Take breaks", "Listen to music", "Seek help from teachers",
    
    # Q6 - Favorite time at home
    "Morning — Fresh start", "Evening — Family time", 
    "Night — Quiet and peaceful", "Afternoon — Productive hours",
    
    # Q7 - Free time activities
    "Reading", "Hanging out with family", "Sleeping/resting", 
    "Watching shows/movies", "Gaming", "Creative hobbies",
    
    # Q8 - Home environment
    "Quiet, and I love it", "Lively, and I enjoy it", 
    "Balanced between quiet and lively",
    
    # Q9 - Who you talk to
    "Parent(s)", "Sibling(s)", "Best friend", "Partner/significant other",
    "Teacher/mentor", "No one",
    
    # Q10 - Memorable support
    "Listened without judgment", "Helped me through a tough time",
    "Encouraged me to pursue a dream", "Celebrated my achievements",
    "Comforted me when I was upset", "Provided practical help",
    
    # Q11 - Mood boosters
    "Listening to music", "Exercising or sports", "Going outdoors/nature", 
    "Talking to loved ones", "Watching movies/shows", "Eating favorite food",
    
    # Q12 - Misunderstandings
    "My quietness", "My seriousness", "My sense of responsibility",
    "My humor", "My work ethic", "My emotions",
    
    # Q13 - Mood changers
    "A kind word", "A small success", "Feeling appreciated",
    "Receiving help", "Positive feedback", "Unexpected kindness",
    
    # Q14 - Career outlook
    "Yes, definitely", "Maybe, still figuring it out", 
    "Not sure, but open to possibilities",
    
    # Q15 - Skills to develop
    "Communication skills", "Leadership skills", 
    "Technical skills (coding, engineering, etc.)",
    "Creative skills (design, writing, art)", "Problem-solving",
    "Time management",
    
    # Q16 - Work-life balance
    "Very important — I need downtime", 
    "I prefer flexible work hours", "Balance is crucial for me",
    "I can adjust as needed",
    
    # Q17 - Supporting a friend
    "Listen without interrupting", "Just be there with them", 
    "Help find solutions", "Distract them with fun activities",
    "Offer practical help", "Encourage professional help",
    
    # Q18 - Advice that stuck
    "\"Stay true to yourself.\"", "\"Don't be afraid to fail.\"", 
    "\"Take things one step at a time.\"", "\"Hard times don't last forever.\"", 
    "\"You are stronger than you think.\"", "\"This too shall pass.\"",
    
    # Q19 - Processing emotions
    "Write them down", "Talk to someone", "Take time alone",
    "Express creatively", "Exercise", "Seek professional help",
    
    # Q20 - What success means
    "Achieving personal goals", "Being happy and content", 
    "Making a difference in others' lives", "Recognition and respect",
    "Financial stability", "Work-life balance"
}

def preprocess_dataset(df):
    """Process the raw dataset into features and labels"""
    # Select relevant columns and clean data
    processed_data = []
    
    for _, row in df.iterrows():
        # Count positive responses for each row
        positive_count = 0
        features = []
        
        # Process each response column (first 20 question columns)
        for col in df.columns[:20]:  # Take first 20 columns only
            response = row[col]
            if pd.isna(response):
                features.append(0)
                continue
                
            # Handle multi-response answers (split by comma)
            responses = [r.strip() for r in str(response).split(",")]
            positive = any(r in POSITIVE_RESPONSES for r in responses)
            features.append(1 if positive else 0)
            positive_count += 1 if positive else 0
            
        # Determine label based on positive count (Firefly logic)
        if positive_count >= 15:
            label = "Healthy (Asymptomatic)"
        elif positive_count >= 8:
            label = "At risk (Subclinical)"
        else:
            label = "Needs Support (Clinical)"
            
        processed_data.append(features + [label])
    
    # Create DataFrame with exactly 20 features + label
    columns = [f"Q{i+1}" for i in range(20)] + ["Label"]
    return pd.DataFrame(processed_data, columns=columns)

def generate_synthetic_data(num_samples=2000):
    """Generate synthetic data with realistic distributions - ensure 20 features"""
    np.random.seed(42)
    X = []
    y = []
    
    for _ in range(num_samples):
        # Generate correlated features
        if np.random.random() < 0.25:  # 25% Clinical
            positive_count = np.random.randint(0, 8)
            label = "Needs Support (Clinical)"
        elif np.random.random() < 0.4:  # 30% Subclinical (40% of remaining 75%)
            positive_count = np.random.randint(8, 15)
            label = "At risk (Subclinical)"
        else:  # 45% Healthy
            positive_count = np.random.randint(15, 21)
            label = "Healthy (Asymptomatic)"
        
        # Ensure exactly 20 features
        features = [1] * positive_count + [0] * (20 - positive_count)
        np.random.shuffle(features)
        X.append(features)
        y.append(label)
    
    return np.array(X), np.array(y)

def firefly_predict(score):
    """Firefly algorithm prediction based on score"""
    if score >= 15:
        return "Healthy (Asymptomatic)"
    elif score >= 8:
        return "At risk (Subclinical)"
    else:
        return "Needs Support (Clinical)"

@st.cache_resource
def train_hybrid_model(df):
    """Train hybrid model with both synthetic and real data"""
    # Process real data - ensure it has exactly 20 features
    processed_df = preprocess_dataset(df)
    X_real = processed_df.iloc[:, :20].values  # Take first 20 columns only
    y_real = processed_df.iloc[:, -1].values
    
    # Generate synthetic data with matching 20 features
    X_synth, y_synth = generate_synthetic_data(1000)
    
    # Verify shapes before concatenation
    if X_real.shape[1] != X_synth.shape[1]:
        raise ValueError(f"Feature dimension mismatch: real data has {X_real.shape[1]} features, synthetic has {X_synth.shape[1]}")
    
    # Combine datasets
    X = np.vstack((X_real, X_synth))
    y = np.concatenate((y_real, y_synth))
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Apply SMOTE for class imbalance
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y_encoded)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )
    
    # Train ANN
    ann = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=1000,
        random_state=42,
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20
    )
    ann.fit(X_train, y_train)
    
    # Convert back to original labels
    y_test_labels = le.inverse_transform(y_test)
    
    return ann, le, X_test, y_test_labels, X_res, y_res

def plot_hybrid_metrics(y_true, y_pred_ann, y_pred_firefly, y_pred_hybrid):
    """Visualize metrics for all three models"""
    plt.style.use('default')
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    
    # Define model names
    models = {
        "ANN": y_pred_ann,
        "Firefly": y_pred_firefly,
        "Hybrid": y_pred_hybrid
    }
    
    # Create plots for each model
    for i, (model_name, y_pred) in enumerate(models.items()):
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=CLASSES, zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
        
        # Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", 
                    xticklabels=CLASSES, yticklabels=CLASSES, 
                    ax=axes[i, 0], cbar=False)
        axes[i, 0].set_title(f'{model_name} - Confusion Matrix', pad=10)
        axes[i, 0].set_xlabel('Predicted')
        axes[i, 0].set_ylabel('Actual')
        
        # Metrics Bar Plot
        metrics_df = pd.DataFrame({
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }, index=CLASSES)
        metrics_df.plot(kind='bar', ax=axes[i, 1], rot=45)
        axes[i, 1].set_title(f'{model_name} - Metrics', pad=10)
        axes[i, 1].set_ylim(0, 1.1)
        axes[i, 1].grid(True, alpha=0.3)
        
        # Metrics Summary
        axes[i, 2].axis('off')
        report = classification_report(y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0)
        summary_text = f"""
        {model_name} Model
        
        Accuracy: {accuracy:.3f}
        
        Class Metrics:
        • Healthy: P={precision[2]:.3f}, R={recall[2]:.3f}, F1={f1[2]:.3f}
        • At Risk: P={precision[1]:.3f}, R={recall[1]:.3f}, F1={f1[1]:.3f}
        • Clinical: P={precision[0]:.3f}, R={recall[0]:.3f}, F1={f1[0]:.3f}
        
        Macro Avg:
        F1: {report['macro avg']['f1-score']:.3f}
        """
        axes[i, 2].text(0.1, 0.5, summary_text, fontsize=10)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display detailed reports
    st.subheader("Detailed Classification Reports")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**ANN Model Report**")
        st.text(classification_report(y_true, y_pred_ann, target_names=CLASSES, zero_division=0))
    with col2:
        st.markdown("**Firefly Algorithm Report**")
        st.text(classification_report(y_true, y_pred_firefly, target_names=CLASSES, zero_division=0))
    with col3:
        st.markdown("**Hybrid Model Report**")
        st.text(classification_report(y_true, y_pred_hybrid, target_names=CLASSES, zero_division=0))

def hybrid_predict(ann, le, firefly_score, features):
    """Combine ANN and Firefly predictions"""
    # ANN prediction
    ann_proba = ann.predict_proba(features.reshape(1, -1))[0]
    ann_pred = le.inverse_transform([np.argmax(ann_proba)])[0]
    
    # Firefly prediction
    firefly_pred = firefly_predict(firefly_score)
    
    # Combine predictions (weighted average)
    class_weights = {
        "Needs Support (Clinical)": 0,
        "At risk (Subclinical)": 1,
        "Healthy (Asymptomatic)": 2
    }
    
    ann_weight = 0.4  # Weight for ANN
    firefly_weight = 0.6  # Weight for Firefly
    
    combined_score = (
        ann_weight * class_weights[ann_pred] + 
        firefly_weight * class_weights[firefly_pred]
    )
    
    # Determine final prediction
    if combined_score >= 1.5:
        return "Healthy (Asymptomatic)", ann_proba
    elif combined_score >= 0.5:
        return "At risk (Subclinical)", ann_proba
    else:
        return "Needs Support (Clinical)", ann_proba

def main():
    st.title("🧠 Hybrid ANN-Firefly Mental Health Predictor")
    st.markdown("---")
    
    # Load and process dataset
    try:
        df = pd.read_excel("dataset.xlsx")
        # Ensure we only take the first 20 question columns
        df = df.iloc[:, :20]  # This ensures we have exactly 20 features
        st.success("✅ Dataset loaded successfully")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()
    
    # Train models
    with st.spinner("Training hybrid model (this may take a minute)..."):
        try:
            ann, le, X_test, y_test, X_res, y_res = train_hybrid_model(df)
            st.success("✅ Model training completed")
        except Exception as e:
            st.error(f"Error training model: {e}")
            st.stop()
    
    # Create tabs
    tab1, tab2 = st.tabs(["📋 Assessment", "📊 Model Evaluation"])
    
    with tab1:
        st.header("Mental Health Assessment")
        
        # Questions and options from the Google Form (20 questions)
        questions = [
            {
                "text": "How do you usually feel during a typical day at your school/college/university?",
                "options": ["Happy", "Motivated", "Stressed", "Tired", "Anxious", "Calm", "Energetic", "Bored"]
            },
            {
                "text": "What's something you really enjoy about your current educational journey?",
                "options": ["Learning new things", "Meeting new people", "Exploring career options", 
                           "Achieving good grades", "Campus environment", "Extracurricular activities", 
                           "None of the above"]
            },
            {
                "text": "If you could change one thing about your daily academic routine, what would it be?",
                "options": ["More practical work", "Better time management", "More support from teachers",
                            "More interactive classes", "Better study materials", "Less workload",
                            "Nothing, it's fine as is"]
            },
            {
                "text": "Do you prefer working alone or in a group when it comes to assignments or projects? Why?",
                "options": ["Group — I enjoy collaboration", "Group — It helps me learn faster", 
                            "Alone — I focus better", "Alone — I work at my own pace",
                            "Depends on the task", "I don't have a preference"]
            },
            {
                "text": "How do you handle stress when exams or deadlines are near?",
                "options": ["Study in advance", "Talk to friends/family", "Meditate or exercise",
                           "Take breaks", "Listen to music", "Seek help from teachers",
                           "Procrastinate", "Panic"]
            },
            {
                "text": "What's your favorite time of day when you're at home, and why?",
                "options": ["Morning — Fresh start", "Evening — Family time", 
                           "Night — Quiet and peaceful", "Afternoon — Productive hours",
                           "No particular preference"]
            },
            {
                "text": "How do you usually spend your weekends or free time at home?",
                "options": ["Reading", "Hanging out with family", "Sleeping/resting", 
                            "Watching shows/movies", "Gaming", "Creative hobbies",
                            "Studying", "Social media"]
            },
            {
                "text": "Is your home more of a quiet space or a lively one? Which do you prefer?",
                "options": ["Quiet, and I love it", "Lively, and I enjoy it", 
                           "Quiet, but I wish it was livelier", "Lively, but I wish it was quieter",
                           "Balanced between quiet and lively"]
            },
            {
                "text": "Who do you usually turn to first when you have something exciting or upsetting to share?",
                "options": ["Parent(s)", "Sibling(s)", "Best friend", "Partner/significant other",
                            "Teacher/mentor", "No one", "Other family member"]
            },
            {
                "text": "What's one memorable thing a family member or close friend did for you that made you feel supported?",
                "options": ["Listened without judgment", "Helped me through a tough time",
                            "Encouraged me to pursue a dream", "Celebrated my achievements",
                            "Comforted me when I was upset", "Provided practical help",
                            "Can't recall any specific instance"]
            },
            {
                "text": "What type of activities instantly lift your mood?",
                "options": ["Listening to music", "Exercising or sports", "Going outdoors/nature", 
                            "Talking to loved ones", "Watching movies/shows", "Eating favorite food",
                            "Shopping", "Nothing really helps"]
            },
            {
                "text": "What's one thing people often misunderstand about you?",
                "options": ["My quietness", "My seriousness", "My sense of responsibility",
                            "My humor", "My work ethic", "My emotions",
                            "Nothing, people understand me well"]
            },
            {
                "text": "What's something small that can make or break your day?",
                "options": ["A kind word", "A small success", "Feeling appreciated",
                            "Receiving help", "Positive feedback", "Unexpected kindness",
                            "Nothing, I stay balanced"]
            },
            {
                "text": "Do you see yourself working in your current field of study long-term?",
                "options": ["Yes, definitely", "Maybe, still figuring it out", 
                            "Not sure, but open to possibilities", "No, planning to switch",
                            "Haven't thought about it"]
            },
            {
                "text": "What's one skill you'd love to develop that could help in your future job?",
                "options": ["Communication skills", "Leadership skills", 
                            "Technical skills (coding, engineering, etc.)",
                            "Creative skills (design, writing, art)", "Problem-solving",
                            "Time management", "Not sure"]
            },
            {
                "text": "How important is work-life balance to you, and what does that look like in your mind?",
                "options": ["Very important — I need downtime", 
                            "I prefer flexible work hours", "Balance is crucial for me",
                            "I can adjust as needed", "Not a priority right now",
                            "Haven't thought about it"]
            },
            {
                "text": "If a friend was going through a tough time, what would you say or do to support them?",
                "options": ["Listen without interrupting", "Just be there with them", 
                            "Help find solutions", "Distract them with fun activities",
                            "Offer practical help", "Encourage professional help",
                            "Not sure how to help"]
            },
            {
                "text": "What's a piece of advice you've received that has stuck with you?",
                "options": ["\"Stay true to yourself.\"", "\"Don't be afraid to fail.\"", 
                            "\"Take things one step at a time.\"", "\"Hard times don't last forever.\"", 
                            "\"You are stronger than you think.\"", "\"This too shall pass.\"",
                            "Can't recall any specific advice"]
            },
            {
                "text": "How do you usually process your thoughts when you're confused or overwhelmed?",
                "options": ["Write them down", "Talk to someone", "Take time alone",
                            "Express creatively", "Exercise", "Seek professional help",
                            "Ignore them until they pass"]
            },
            {
                "text": "What does 'success' mean to you personally?",
                "options": ["Achieving personal goals", "Being happy and content", 
                            "Making a difference in others' lives", "Recognition and respect",
                            "Financial stability", "Work-life balance",
                            "Not sure yet"]
            }
        ]
        
        responses = []
        cols = st.columns(2)
        
        for i, question in enumerate(questions):
            with cols[i % 2]:
                st.markdown(f"**Q{i+1}: {question['text']}**")
                response = st.multiselect(
                    "Select one or more options:",
                    options=question["options"],
                    key=f"q{i+1}"
                )
                responses.append(", ".join(response) if response else "")
        
        if st.button("🔍 Predict Mental Health Stage", type="primary"):
            if not all(responses):
                st.warning("⚠️ Please answer all questions before submitting.")
                st.stop()
            
            # Process responses
            features = []
            positive_count = 0
            
            for response in responses:
                # Handle multi-response answers
                responses_list = [r.strip() for r in str(response).split(",")]
                positive = any(r in POSITIVE_RESPONSES for r in responses_list)
                features.append(1 if positive else 0)
                positive_count += 1 if positive else 0
            
            features = np.array(features)
            
            # Get predictions
            firefly_pred = firefly_predict(positive_count)
            hybrid_pred, hybrid_proba = hybrid_predict(ann, le, positive_count, features)
            
            # Display results
            st.success("### 📈 Analysis Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("💡 Positive Responses", f"{positive_count}/20")
            with col2:
                st.metric("🔥 Firefly Prediction", firefly_pred)
            with col3:
                st.metric("🤖 Hybrid Prediction", hybrid_pred)
                st.caption(f"Confidence: {max(hybrid_proba):.1%}")
            
            # Show probabilities
            st.subheader("🎯 Prediction Probabilities")
            prob_df = pd.DataFrame({
                'Mental Health Stage': CLASSES,
                'Probability': hybrid_proba,
                'Percentage': [f"{p:.1%}" for p in hybrid_proba]
            })
            st.dataframe(prob_df, use_container_width=True)
    
    with tab2:
        st.header("📊 Model Performance Evaluation")
        
        if st.button("Generate Evaluation Metrics", type="primary"):
            with st.spinner("Evaluating models..."):
                # Get predictions for all models
                y_pred_ann = le.inverse_transform(ann.predict(X_test))
                y_pred_firefly = [firefly_predict(sum(x)) for x in X_test]
                
                # Hybrid predictions
                y_pred_hybrid = []
                for i, x in enumerate(X_test):
                    score = sum(x)
                    pred, _ = hybrid_predict(ann, le, score, x)
                    y_pred_hybrid.append(pred)
                
                # Plot metrics
                st.subheader("Model Comparison Metrics")
                plot_hybrid_metrics(y_test, y_pred_ann, y_pred_firefly, y_pred_hybrid)
                
                # Show class distribution after SMOTE
                st.subheader("Class Distribution After SMOTE")
                class_counts = Counter(le.inverse_transform(y_res))
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(class_counts.keys(), class_counts.values(), color=['#ff7f7f', '#ffb347', '#90ee90'])
                ax.set_title("Class Distribution After SMOTE")
                ax.set_ylabel("Number of Samples")
                st.pyplot(fig)
    
    # Sidebar information
    with st.sidebar:
        st.header("ℹ️ About This Tool")
        
        st.markdown("""
        This hybrid model combines:
        - **Artificial Neural Network (ANN)**: Deep learning model trained on survey responses
        - **Firefly Algorithm**: Rule-based scoring system
        
        **Mental Health Categories**:
        - 🟢 Healthy (Asymptomatic): No significant concerns
        - 🟡 At Risk (Subclinical): Some concerning signs
        - 🔴 Needs Support (Clinical): May benefit from professional support
        """)
        
        st.markdown("---")
        st.markdown("**Dataset Statistics**")
        st.write(f"- Total responses: {len(df)}")
        st.write("- Features: 20 survey questions")
        
        st.markdown("---")
        st.markdown("**Model Architecture**")
        st.write("- ANN: 2 hidden layers (64 → 32 neurons)")
        st.write("- SMOTE: Applied to handle class imbalance")
        st.write("- Hybrid Weighting: 40% ANN, 60% Firefly")

if __name__ == "__main__":
    main()