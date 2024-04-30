from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd

from pkg.utils import loadObj

app = FastAPI()

class ScoringItem(BaseModel):
    pct_tl_open_L6M:  float
    pct_tl_closed_L6M: float
    Tot_TL_closed_L12M: int  
    pct_tl_closed_L12M: float
    Tot_Missed_Pmnt: int  
    CC_TL: int  
    Home_TL: int  
    PL_TL: int  
    Secured_TL: int  
    Unsecured_TL: int  
    Other_TL: int  
    Age_Oldest_TL: int  
    Age_Newest_TL: int  
    time_since_recent_payment: int  
    max_recent_level_of_deliq: int  
    num_deliq_6_12mts: int  
    num_times_60p_dpd: int  
    num_std_12mts: int  
    num_sub:int  
    num_sub_6mts:int  
    num_sub_12mts:int  
    num_dbt: int  
    num_dbt_12mts: int  
    num_lss: int 
    recent_level_of_deliq: int  
    CC_enq_L12m: int  
    PL_enq_L12m: int  
    time_since_recent_enq: int
    enq_L3m: int 
    NETMONTHLYINCOME: int  
    Time_With_Curr_Empr: int  
    CC_Flag: int  
    PL_Flag: int  
    pct_PL_enq_L6m_of_ever: float
    pct_CC_enq_L6m_of_ever: float
    HL_Flag: int  
    GL_Flag: int  
    EDUCATION: int  
    MARITALSTATUS_Married: int  
    MARITALSTATUS_Single: int  
    GENDER_F: int  
    GENDER_M: int  
    last_prod_enq2_AL: int  
    last_prod_enq2_CC: int  
    last_prod_enq2_ConsumerLoan: int  
    last_prod_enq2_HL: int  
    last_prod_enq2_PL: int  
    last_prod_enq2_others: int
    first_prod_enq2_AL: int  
    first_prod_enq2_CC: int   
    first_prod_enq2_ConsumerLoan:int  
    first_prod_enq2_HL: int  
    first_prod_enq2_PL: int  
    first_prod_enq2_others: int  

model = loadObj('../models/model.pkl')

@app.post('/')
async def scoring_endpoint(item: ScoringItem):
    itemDict = item.dict()
    df = pd.DataFrame([itemDict.values()], columns= itemDict.keys())
    ypred = model.predict(df)

    return {"prediction": int(ypred)}