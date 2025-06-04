from dotenv import load_dotenv
import os
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.llms import OpenAI

# Load environment variables
load_dotenv()

print("MYSQL_USER:", os.getenv('MYSQL_USER'))
print("MYSQL_PWD:", os.getenv('MYSQL_PWD'))
print("MYSQL_HOST:", os.getenv('MYSQL_HOST'))
print("MYSQL_DB:", os.getenv('MYSQL_DB'))



# MySQL connection string
DB_URI = f"mysql+pymysql://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PWD')}@{os.getenv('MYSQL_HOST')}/{os.getenv('MYSQL_DB')}"
print("DB_URI:", DB_URI)
engine = create_engine(DB_URI)
db = SQLDatabase(engine)

# Initialize LLM
llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.0)

# Finance-specific prompt
SYSTEM_PROMPT = """
You are a financial data analyst expert in MySQL.
Given a database schema and a natural language question about finance,
generate an accurate SQL query to answer the question.
Return only the SQL query with comments explaining each part.
Use correct table and column names from the schema.
Do not add LIMIT unless explicitly requested.
Schema:
- clients: (client_id, first_name, last_name, company, address, city, state, country, postal_code, phone, email, created_at)
- employees: (employee_id, first_name, last_name, title, reports_to, hire_date, phone, email)
- currencies: (currency_id, currency_code, currency_name, country)
- financial_products: (product_id, product_code, product_name, base_rate_bps, spread_bps, effective_dt)
- fx_transactions: (transaction_id, client_id, product_id, currency_pair, amount_usd, exchange_rate, trade_time, trade_type, status, employee_id)
- portfolio_balances: (balance_id, client_id, currency_id, balance_amount, last_updated)
- transaction_fees: (fee_id, transaction_id, fee_type, fee_amount, fee_date)
"""

# Initialize SQLDatabaseChain
sql_chain = SQLDatabaseChain.from_llm(
    llm=llm,
    db=db,
    prompt=SYSTEM_PROMPT,
    verbose=True
)

# Integrate with RAG system
def financial_qa(question: str, rag_response: str = None):
    try:
        sql_result = sql_chain.run(question)
        combined_response = f"SQL Result: {sql_result}"
        if rag_response:
            combined_response += f"\nRAG Context: {rag_response}"
        return combined_response
    except Exception as e:
        return f"Error: {str(e)}"

# Test the system
if __name__ == "__main__":
    question = "Show the total USD amount of FX trades for John Doe in May 2024."
    rag_response = "John Doe is a client of GlobalCorp with significant FX trading activity."
    result = financial_qa(question, rag_response)
    print("Combined Response:", result)