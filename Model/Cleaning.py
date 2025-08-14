import pandas as pd
df=pd.read_csv('survey.csv')
df['self_employed']=df['self_employed'].fillna('Dont know')
df['work_interfere']=df['work_interfere'].fillna('Dont know')
df["Gender"] = df["Gender"].apply(lambda x: "Male" if str(x).strip().lower() in ['male', 'man', 'm', 'cis male', 'cis man', 'mail'] else"Female" if str(x).strip().lower() in ['female', 'woman', 'cis woman', 'cis female'] else"Other")
df['Timestamp']=pd.to_datetime(df['Timestamp'])
df.drop(columns=['Timestamp','comments','state'],inplace=True)
df = df[(df["Age"] >= 16) & (df["Age"] <= 100)]
def clean(row) :
    cleaned_row=[]
    for i in row:
        i=i.strip().lower()
        if '-' in i:
            newi=i.split('-')
            if len(newi)==2 and newi[0].isdigit() and newi[1].isdigit():
                cleaned_row.append(int((int(newi[0])+int(newi[1]))/2))
            else:
                cleaned_row.append('Unknown')
        elif i.strip().isdigit():
            cleaned_row.append(int(i.strip()))
        elif "more than" in i:
           x= int(i[(i.index('more than')+len('more than')):].strip())
           cleaned_row.append(x+1)
        else:cleaned_row.append("Unknown")
    return cleaned_row
new_no_employees=clean(df['no_employees'])
df['no_employees']=new_no_employees
df.to_csv("clean_data.csv", index=False)
