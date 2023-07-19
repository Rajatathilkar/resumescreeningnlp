
import numpy as np 
import streamlit as st 
import pickle 
import re 
import nltk 
nltk.download('punkt')
nltk.download('stopwords')
clf=pickle.load(open('clf.pkl','rb'))
tfidf=pickle.load(open('tfidf.pkl','rb'))
def cleanresume(txt):
    cleantxt=re.sub('http\S+\s',' ',txt)
    cleantxt=re.sub('RT|cc',' ',cleantxt)
    cleantxt=re.sub('@\S+', ' ',cleantxt)
    cleantxt=re.sub('#\S+\s',' ',cleantxt)
    cleantxt=re.sub('[%s]' % re.escape("""|"#$%&'()*+,-,/:;<=>?@[\]^_'{|}~"""),' ',cleantxt)
   # cleantxt=re.sub('\s+',' ',cleantxt)
    #cleantxt=re.sub(r'[^\x00-\x7f]', ' ',cleantxt)
    cleantxt=re.sub(r'[^\x00-\x7f]',' ',cleantxt)
    cleantxt=re.sub('\s+',' ',cleantxt)
    return cleantxt

def main():
    st.title("Resume Screeing App")
    upload_file=st.file('upload resume',type=['txt','pdf'])
    
    
    if upload_file is not None:
        try:
            resume_bytes=uploaded_file.read()
            resume_text=resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_txt=resume_bytes.decode('latin-1')
        cleaned_resume=clean_resume([resume_text])
        cleaned_resume=tfidf.transform([cleaned_resume])
        prediction_id=clf.predict(cleaned_resume)[0]
        st.write(prediction_id)
        category_mapping={'Data Science':6, 'HR':12, 'Advocate':0, 'Arts':1, 'Web Designing':24,
        'Mechanical Engineer':16, 'Sales':22, 'Health and fitness':14,
        'Civil Engineer':5, 'Java Developer':15, 'Business Analyst':4,
         'SAP Developer':21, 'Automation Testing':2, 'Electrical Engineering':11,
         'Operations Manager':18, 'Python Developer':20, 'DevOps Engineer':8,
        'Network Security Engineer':17, 'PMO':19, 'Database':7, 'Hadoop':13,
        'ETL Developer':10, 
         'DotNet Developer':9, 'Blockchain':3, 'Testing':23,
                         }
        category_name=category_mapping.get(prediction_id,'unknown')
        st.write('predicted category:',category_name)
 
 
if __name__="__main__":
    main()
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
