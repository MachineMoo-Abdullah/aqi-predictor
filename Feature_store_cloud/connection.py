from sqlalchemy import create_engine
import os
from pathlib import Path
from dotenv import load_dotenv

dotenv_path = Path(__file__).resolve().parent.parent / ".env"

load_dotenv(dotenv_path)

HOST = os.getenv("SUPABASE_HOST")
DB = os.getenv("SUPABASE_DB")
USER = os.getenv("SUPABASE_USER")
PASSWORD = os.getenv("SUPABASE_PASSWORD")
PORT = os.getenv("SUPABASE_PORT")

DATABASE_URL = (
    f"postgresql://{USER}:{PASSWORD}"
    f"@{HOST}:{PORT}/{DB}"
)
engine = create_engine(DATABASE_URL)