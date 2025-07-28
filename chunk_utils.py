# chunk_utils.py
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

def character_split(text, chunk_size=100, chunk_overlap=20):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def recursive_split(text, chunk_size=100, chunk_overlap=20):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)
