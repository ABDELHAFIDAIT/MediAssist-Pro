from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from app.core.config import settings


engine = create_engine(settings.DATABASE_URL)

session = sessionmaker(bind=engine, autoflush=False, autocommit=False)

Base = declarative_base()