import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from langchain.prompts import PromptTemplate
from Bio import Entrez
import time

# Add your email for NCBI
Entrez.email = "your@email.com"

def format_gene_stats(gene_data: pd.Series) -> str:
    """
    Format gene statistics into a readable string
    """
    stats = {
        'p-value': gene_data.get('pval', 'N/A'),
        'FDR': gene_data.get('fdr', 'N/A'),
        'log2FC': gene_data.get('log2FC', 'N/A')
    }
    
    # Format numbers to be more readable
    for key, value in stats.items():
        if isinstance(value, (float, np.float64)):
            if key in ['pval', 'fdr']:
                stats[key] = f"{value:.2e}" if value < 0.01 else f"{value:.4f}"
            else:
                stats[key] = f"{value:.4f}"
    
    return f"""
Statistical Analysis:
- p-value: {stats['p-value']}
- FDR: {stats['FDR']}
- log2FC: {stats['log2FC']}
"""

def process_experimental_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Process and clean the experimental data
    """
    # Remove duplicates
    data = data.drop_duplicates()
    
    # Handle missing values
    data = data.fillna(method='ffill')
    
    return data

def generate_data_summary(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a summary of the dataset
    """
    summary = {
        "total_rows": len(data),
        "total_columns": len(data.columns),
        "column_types": data.dtypes.to_dict(),
        "missing_values": data.isnull().sum().to_dict(),
        "numeric_columns": data.select_dtypes(include=[np.number]).columns.tolist()
    }
    return summary

def search_pubmed(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search PubMed for relevant articles
    """
    try:
        # Add date sort and recent date filter to the query
        query = f"{query} AND 2020:2024[pdat]"
        
        # Search PubMed
        handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_results,
            sort="date"  # Sort by publication date
        )
        record = Entrez.read(handle)
        handle.close()

        # Get details for each article
        articles = []
        for id in record["IdList"]:
            time.sleep(0.5)  # Be nice to NCBI servers
            handle = Entrez.efetch(db="pubmed", id=id, rettype="medline", retmode="text")
            article = handle.read()
            handle.close()
            
            # Extract basic information
            title = ""
            abstract = ""
            authors = ""
            year = ""
            
            for line in article.split('\n'):
                if line.startswith("TI  -"):
                    title = line[6:].strip()
                elif line.startswith("AB  -"):
                    abstract = line[6:].strip()
                elif line.startswith("AU  -"):
                    authors += line[6:].strip() + ", "
                elif line.startswith("DP  -"):
                    year = line[6:10]
            
            articles.append({
                "title": title,
                "abstract": abstract[:300] + "..." if len(abstract) > 300 else abstract,
                "authors": authors[:-2] if authors else "",
                "year": year,
                "pubmed_id": id,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{id}/"
            })
            
        return articles
    except Exception as e:
        print(f"Error searching PubMed: {str(e)}")
        return []

# Custom prompt template for drug discovery with PubMed context
DRUG_DISCOVERY_PROMPT = PromptTemplate(
    input_variables=["context", "pubmed_context", "question"],
    template="""You are an AI assistant specialized in drug discovery and biological data analysis. 
    Use the following context about the experimental data and relevant PubMed articles to answer the question.
    
    When referring to research papers, ALWAYS use citation format with clickable links. For example:
    - "According to [Smith et al. (2023)](https://pubmed.ncbi.nlm.nih.gov/XXX), the protein..."
    - "Recent research by [Zhang and Li (2024)](https://pubmed.ncbi.nlm.nih.gov/YYY) showed..."
    
    Experimental Data Context:
    {context}
    
    Relevant PubMed Literature:
    {pubmed_context}
    
    Question: {question}
    
    Please provide a detailed yet concise answer based on both the experimental data and scientific literature. Remember to include clickable citation links when referencing papers:"""
)

def prepare_context(data: pd.DataFrame, question: str, include_pubmed: bool = True) -> Dict[str, str]:
    """
    Prepare context for the LLM based on the question, data, and PubMed results
    """
    summary = generate_data_summary(data)
    data_context = f"""
    Dataset Information:
    - Total samples: {summary['total_rows']}
    - Features: {', '.join(data.columns)}
    - Numeric features: {', '.join(summary['numeric_columns'])}
    """
    
    # Add statistical information if the question is about a specific gene
    gene_stats = ""
    for word in question.split():
        if "DMR" in word.upper() or "AT" in word.upper():
            matches = data[data['Fasta_headers'].str.contains(word, case=False, na=False)]
            if not matches.empty:
                gene_stats = format_gene_stats(matches.iloc[0])
                data_context += f"\nGene Statistics for {word}:\n{gene_stats}"
                break
    
    pubmed_context = ""
    if include_pubmed:
        articles = search_pubmed(question)
        if articles:
            pubmed_context = "Recent relevant publications:\n\n"
            for article in articles:
                # Process authors list
                authors = article['authors'].split(', ') if article['authors'] else ["Unknown"]
                
                # Format citation for display
                if len(authors) > 1:
                    display_citation = f"{authors[0]} et al. ({article['year']})"
                else:
                    display_citation = f"{authors[0]} ({article['year']})"
                
                # Format full author list for link
                link_citation = article['authors'].replace(', ', '_').replace(' ', '_') if article['authors'] else "Unknown"
                link_citation = f"{link_citation}_{article['year']}"
                
                pubmed_context += f"[{display_citation}]({article['url']})\n"
                pubmed_context += f"Title: {article['title']}\n"
                pubmed_context += f"Authors: {article['authors']}\n"
                pubmed_context += f"Abstract: {article['abstract']}\n\n"
    
    return {
        "context": data_context,
        "pubmed_context": pubmed_context
    } 