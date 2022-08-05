import en_ner_bc5cdr_md
import spacy
from spacy import displacy
import streamlit as st
import pandas as pd 
import pickle
from flair.data import Sentence
from flair.models import SequenceTagger
import regex as re

#### Defining Functions
def listToString(s): 
    str1 = " ; " 
    return (str1.join(s))

### Defining List
lst_chk = ["PHI Masking","Disease & Drug Extraction","Question Answering"]

sent_lst = ["Doctor in OHIO helped the covid patients to recover quickly"
"The patient has committed suicide due to high stress",
"Indian Research scientist found covaxin and saved all people",
"Students are affected with covid and are expected to die soon",
"Having high fever and not able to wake up"]

st.set_page_config(page_title="My WebPage", page_icon=":hugs:")



hom =st.sidebar.radio("Text Predictions",["Home","Text Prediction Models","Feedback"])


### Main Code
if hom == "Home":
    #colm1 , colm2, colm3, colm4, colm5 = st.columns(5)
    #with colm5:
    #    st.image("data//Goofy.jpg", width = 100)
        
    col1 , col2 = st.columns([0.8,3])
    with col2:
        st.title("GOOFY THE ASSIST :male-doctor:")
        # :male-doctor:,  :person_doing_cartwheel:
    #with col3:
        #st.write("[Gavs](https://www.gavstech.com/)")
    st.caption("")
    st.video("https://www.youtube.com/watch?v=_io1BzRwdWc")
    

if hom == "Text Prediction Models":
    st.sidebar.caption("")
    rad =st.sidebar.radio("Text Prediction Model",lst_chk)

##### PHI Masking        
    if rad == "PHI Masking":
        
        phi_masking_input = ["DR. RAMANUJAM FROM AMAR LEELA HOSPITALS, HYDERABAD REQUIRES A PACK OF SURGICAL PROCEDURE COVER. MAIL RAMANUJAM at ramanujam@gmail.com",
"POLYDIOXANONE MONOFILAMENT SUTURE PRESCRIBED PATIENT NAME PHILIPS ADMITTED AT APEX HOSPITALS, JAIPUR",
"SURGICAL KIT PACKED DELIVERED TO ASIAN HEART INSTITUTE ON 12/11/21",
"FORMER PRESIDENT OF INDIA ZAKIR HUSAIN ADMITTED AT CITY HOSPITAL, DELHI DATED 03-MAY-1969",
"HOSPITAL RELATED QUERIES CAN BE SENT TO CHORDHOSP@GMAIL.COM"]

        st.header("PHI Masking")
        
        sb1 = st.radio("Choose Data",["Choose From Sample","Upload Data"],index = 0)
        
        if sb1 == "Choose From Sample": 
            sb2 = st.selectbox("Select Input Data",phi_masking_input,index = 0)
            ip_txt = sb2
            if st.button("Clear"):
                ip_txt = ""
            txt_input1 = st.text_area("Your text input is ",ip_txt)            
            if st.button("Submit"):

##### Hugging Face Model
              lst_org = []
              lst_date = []
              lst_person = []
              lst_gpe = []

              tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")
              sentence = Sentence(txt_input1)
              tagger.predict(sentence)

              out_txt = txt_input1
              for entity in sentence.get_spans('ner'):
                str_len = len(str(entity.text)) 
                out_txt = out_txt.replace(str(entity.text), "*"*str_len)
                if entity.tag == "ORG":
                  lst_org.append(entity.text)
                elif entity.tag == "DATE":
                  lst_date.append(entity.text)
                elif entity.tag == "PERSON":
                  lst_person.append(entity.text)
                elif entity.tag == "GPE":
                  lst_gpe.append(entity.text)

              lst_mail = re.findall("([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", txt_input1)
              for mail in lst_mail:
                str_len = len(mail)
                out_txt = out_txt.replace(str(mail), "*"*str_len)

              details = {
                  'TEXT' : txt_input1,
                  'ORG' : listToString(lst_org),
                  'DATE' : listToString(lst_date),
                  'PERSON' : listToString(lst_person),
                  'LOCATION' : listToString(lst_gpe),
                  'MAIL' : listToString(lst_mail)}
              phi_df = pd.DataFrame([details])

              st.text_area("Output",out_txt)
              st.table(phi_df)
              st.success("The output has been generated")
                
            
        if sb1 == "Upload Data":
            img = st.file_uploader("Upload a file")
            if img:
                df = pd.read_excel(img,sheet_name="PHI_Masking")
                st.table(df)
                if st.button("Submit"):

#### Hugging Face Model full dataframe
                  phi_df_full = pd.DataFrame()
                  for row in df.itertuples():
                    txt_input1 = str(row.PHI_DESCRIPTION)
                    lst_org = []
                    lst_date = []
                    lst_person = []
                    lst_gpe = []

                    tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")
                    sentence = Sentence(str(row.PHI_DESCRIPTION))
                    tagger.predict(sentence)

                    out_txt = txt_input1

                    for entity in sentence.get_spans('ner'):
                      str_len = len(str(entity.text)) 
                      out_txt = out_txt.replace(str(entity.text), "*"*str_len)
                      if entity.tag == "ORG":
                        lst_org.append(entity.text)
                      elif entity.tag == "DATE":
                        lst_date.append(entity.text)
                      elif entity.tag == "PERSON":
                        lst_person.append(entity.text)
                      elif entity.tag == "GPE":
                        lst_gpe.append(entity.text)

                    lst_mail = re.findall("([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", txt_input1)
                    for mail in lst_mail:
                      str_len = len(mail)
                      out_txt = out_txt.replace(str(mail), "*"*str_len)

                    details = {
                        'TEXT' : txt_input1,
                        'MASKED_TEXT' : out_txt,
                        'ORG' : listToString(lst_org),
                        'DATE' : listToString(lst_date),
                        'PERSON' : listToString(lst_person),
                        'LOCATION' : listToString(lst_gpe),
                        'MAIL' : listToString(lst_mail)}
                    #phi_df_full = phi_df_full.append(details, ignore_index=True)
                    phi_df = pd.DataFrame([details])
                    st.table(phi_df)
                  st.success("The output has been generated")  


##### Disease & Drug Extraction 

    if rad == "Disease & Drug Extraction":
        
        phi_drug = ["Mr Mital has a history of medical conditions. He has had hypertension and hyperlipidemia since 1990 and suffered several strokes in 2005. He subsequently developed heart problems (cardiomyopathy), cardiac failure and chronic renal disease and was treated in Betroth Hospital. He was last admitted to the Betroth Hospital on 1 April 2010 till 15 April 2010, during which he was diagnosed to have suffered from a stroke. This was confirmed by CT and MRI brain scans. Thereafter, he was transferred to Ximz Hospital for stroke rehabilitation on 15 April 2010. After that, Mr Mital was referred to Blackacre Hospital for follow-up treatment from in November 2010. The clinical impression was that he was manifesting behavioural and psychological symptoms secondary to Dementia. Mr Mital’s dementia and stroke have impaired the functioning of his mind and brain. His failure to remember where he was (i.e. in the hospital) and the day and date, despite being told a short while ago, shows his inability to retain information. He was also not able to remember basic information such as his age, and the address where he lives.",
        "This patient is diagnosed with Amnesia. It is a deficit in memory caused by brain damage or disease, but it can also be caused temporarily using various sedatives and hypnotic drugs. The memory can be either wholly or partially lost due to the extent of damage that was caused. The patient has Anterograde amnesia, the inability to transfer new information from the short-term store into the long-term store. People with anterograde amnesia cannot remember things for long periods of time. These two types are not mutually exclusive; both can occur simultaneously. Case studies also show that amnesia is typically associated with damage to the medial temporal lobe. Reports shows that the areas of the diencephalon are damaged, a correlation between deficiency of RbAp48 protein and memory loss. A severe reduction in the ability to learn new material and retrieve old information is observed. ",
        "Mr.D is diagnosed with Arthritis, joint pain and stiffness, which typically worsen with age. Reports shows a sign of Osteoarthritis, slippery tissue that covers the ends of bones where they form a joint to break down, changes in the bones and deterioration of the connective tissues that attach muscle to bone and hold the joint together. Physical examination on joints for swelling, redness and warmth and obtained samples of blood, urine and joint fluid. Medications prescribed for treatment are  NSAIDs. like ibuprofen Advil, Motrin IB and naproxen sodium (Aleve).",
        "Reports declare the patient has Tuberculosis (TB), a potentially serious infectious disease that mainly affects the lungs. Symptoms include coughing for three or more weeks, coughing up blood or mucus, chest pain, unintentional weight loss, fatigue, fever, loss of appetite. Body parts affected including the kidneys and spine. Diagnostic tool used - skin test. Medications prescribed Isoniazid, Rifampin (Rifadin, Rimactane), Ethambutol (Myambutol), Pyrazinamide, a combination of antibiotics called fluoroquinolones and injectable medications, such as amikacin or capreomycin (Capastat).",
        "Disease detected – Malaria, spread from the bite of infected mosquitoes. Symptoms on report - high fever and shaking chills, general discomfort, Headache, Nausea and vomiting, Diarrhea, Abdominal pain, Muscle or joint pain, Fatigue, Rapid breathing, Rapid heart rate, Cough. Antimalarial drugs prescribed - Chloroquine phosphate. Chloroquine is the preferred treatment for any parasite that is sensitive to the drug. Artemisinin-based combination therapies (ACTs), artemether-lumefantrine (Coartem) and artesunate-mefloquine."]

        st.header("Disease & Drug Extraction")
        
        sb1 = st.radio("Choose Data",["Choose From Sample","Upload Data"],index = 0)
        
        if sb1 == "Choose From Sample": 
            sb2 = st.selectbox("Select Input Data",phi_drug,index = 0)
            ip_txt = sb2
            if st.button("Clear"):
                ip_txt = ""
            txt_input1 = st.text_area("Your text input is ",ip_txt)            
            if st.button("Submit"):

##### Scispacy Model
              nlp_bc5cdr = en_ner_bc5cdr_md.load()
              doc_nlp_bc5cdr = nlp_bc5cdr(txt_input1)
              #displacy_image = displacy.render(doc_nlp_bc5cdr, jupyter=True,style='ent')
              lst_disease = []
              lst_chemical = []
              
              for ent in doc_nlp_bc5cdr.ents:
                if ent.label_ == "DISEASE":
                  lst_disease.append(str(ent.text))
                elif ent.label_ == "CHEMICAL":
                  lst_chemical.append(str(ent.text))
              lst_disease = list(set(lst_disease))
              lst_chemical = list(set(lst_chemical))
              
              details = {
                  'TEXT' : txt_input1,
                  'Disease' : listToString(lst_disease),
                  'Chemical' : listToString(lst_chemical)}

              #phi_df_full = phi_df_full.append(details, ignore_index=True)
              df = pd.DataFrame([details])

              #st.image(displacy_image)
              st.table(df)
              st.success("The output has been generated")
                
            
        if sb1 == "Upload Data":
            img = st.file_uploader("Upload a file")
            if img:
                df = pd.read_excel(img,sheet_name="Disease_Drug_Extraction")
                st.table(df)
                if st.button("Submit"):

#### Scispacy Model full dataframe
                  phi_df_full = pd.DataFrame()
                  for row in df.itertuples():
                    nlp_bc5cdr = en_ner_bc5cdr_md.load()
                    doc_nlp_bc5cdr = nlp_bc5cdr(str(row.Medical_Record))
                    #displacy_image = displacy.render(doc_nlp_bc5cdr, jupyter=True,style='ent')
                    lst_disease = []
                    lst_chemical = []
                    
                    for ent in doc_nlp_bc5cdr.ents:
                      if ent.label_ == "DISEASE":
                        lst_disease.append(str(ent.text))
                      elif ent.label_ == "CHEMICAL":
                        lst_chemical.append(str(ent.text))
                    lst_disease = list(set(lst_disease))
                    lst_chemical = list(set(lst_chemical))
                    
                    details = {
                        'TEXT' : str(row.Medical_Record),
                        'Disease' : listToString(lst_disease),
                        'Chemical' : listToString(lst_chemical)}

                    #phi_df_full = phi_df_full.append(details, ignore_index=True)
                    phi_df = pd.DataFrame([details])
                    st.table(phi_df)
                  st.success("The output has been generated")


##### Question Answering 
    if rad == "Question Answering":
        str_txt = "Mr.Mark Jade is diagnosed with Arthritis, joint pain and stiffness, which typically worsen with age. Reports shows a sign of Osteoarthritis, slippery tissue that covers the ends of bones where they form a joint to break down, changes in the bones and deterioration of the connective tissues that attach muscle to bone and hold the joint together. Physical examination on joints for swelling, redness and warmth and obtained samples of blood, urine and joint fluid. Medications prescribed for treatment are  NSAIDs. like ibuprofen Advil, Motrin IB and naproxen sodium (Aleve)."
        questions = ["What is the patient name?",
                      "The patient is diagnosed with?",
                      "What does the report say?",
                      "What all are the samples obtained?",
                      "What medications are prescribed?"]

        st.header("Question Answering")
        context = st.text_area("Your text input is ",str_txt,)   
        qstn = st.selectbox("Select Question",questions,index = 0)

        if st.button("Submit"):
            from transformers import AutoTokenizer
            import transformers
            from transformers import pipeline
            qa_model = pipeline("question-answering")
            qa_op = qa_model(question = qstn, context = context)

            cnt = 0
            for i in qa_op.values():
              cnt += 1
              if cnt == 4:
                op_fin = str(i)

            #st.write("Output",op_fin)
            st.success(op_fin)
                
        

#### Feedback

if hom == "Feedback":
    col3 , col4 = st.columns([1,4])
    with col4:
        st.subheader("Do let us know your Feedback :sunglasses:")

    st.caption("")
    name = st.text_input("Enter your Name")
    feedback = st.text_area("Enter your Feedback")
    if st.button("Submit"):
        to_add = {"Name":[name],"Feedback":[feedback]}
        to_add = pd.DataFrame(to_add)
        to_add.to_csv("data//feedback.csv",mode='a',header = False,index= False)
        st.success("Submitted")
        st.caption("")
    
    col5 , col6 = st.columns([1,4])
    with col6:
        st.caption("")
        if st.checkbox("View Previous Feedback"): 
            fb_csv = pd.read_csv("data//Feedback.csv")
            st.table(fb_csv)