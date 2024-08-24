from qdrant_client import QdrantClient
from qdrant_client import QdrantClient, models
import pandas as pd
from fastembed import SparseTextEmbedding, TextEmbedding
import os
import numpy as np






model_bm42 = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
model_jina = TextEmbedding(model_name="jinaai/jina-embeddings-v2-base-en")

client = QdrantClient(host='localhost', port=6333)


def create_database():
    client.create_collection(
    collection_name="my-hybrid-collection",
    vectors_config={
        "jina": models.VectorParams(
            size=768,
            distance=models.Distance.COSINE,
        )
    },
    sparse_vectors_config={
        "bm42": models.SparseVectorParams(
            modifier=models.Modifier.IDF,
        )
    }
    )


def upload_data_to_qdrant():
    try:
        data_csv_path= os.path.join('myapp','large_social_posts.csv')
        print('reading completed')
        df = pd.read_csv(data_csv_path)
        df=df.head(50)
    
        df['content2']= df['title'] + ' ' +  df['content'] + ' ' +  df['tags'] 

        texts = df['content2'].tolist()
        
      
        print("Starting embedding process")

        sparse_embeddings = []
        dense_embeddings = []

        for text in texts:           
 # Convert generators to lists
            sparse_embedding_gen = model_bm42.query_embed(text)
            dense_embedding_gen = model_jina.query_embed(text)

            # Retrieve the first item from the generator
            sparse_embedding = next(sparse_embedding_gen)
            dense_embedding = next(dense_embedding_gen)


            if hasattr(sparse_embedding, 'values') and hasattr(sparse_embedding, 'indices'):
                sparse_embedding_list = sparse_embedding.values.tolist()
                sparse_indices_list = sparse_embedding.indices.tolist()
                sparse_embeddings.append((sparse_indices_list, sparse_embedding_list))
            else:
                raise ValueError("Unexpected sparse embedding format")

            if isinstance(dense_embedding, np.ndarray):
                dense_embeddings.append(dense_embedding.tolist())
            else:
                raise ValueError("Unexpected dense embedding format")




        collection_name = 'my-hybrid-collection'  

        points = [
            {
                "id": idx,
                "vector": {
                    "jina": dense_embeddings[idx],
                    "bm42": {"values": sparse_embeddings[idx][1], "indices": sparse_embeddings[idx][0]}
                },
                 "payload":{
                "title": df.loc[idx,"title"],
                "tag": df.loc[idx,"tags"],
            }
            }
            for idx in range(len(texts))
        ]

        response = client.upsert(collection_name=collection_name, points=points)
        print(f"Upsert response: {response}")

        if response is not None:
           return response
        else:
            return None
    except Exception as e:
        print(f'Error occured : {e}')
        return None





def search_query(query_text,query_keyword):
    try:
        sparse_embedding = list(model_bm42.query_embed(query_keyword))[0]
        dense_embedding = list(model_jina.query_embed(query_text))[0]

        search_results = client.query_points(
            collection_name='my-hybrid-collection',
            prefetch=[
                models.Prefetch(query=sparse_embedding.as_object(), using="bm42", limit=10),
             
                models.Prefetch(query=dense_embedding.tolist(), using="jina", limit=10),
            ],
           
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=10
        )
        
        print("search result")
        points=search_results.points
        print(points)
  
       

        return [{"id": result.id, "score": result.score , "payload": result.payload} for result in points]
    except Exception as e:
        print(f'Error occured : {e}')
  



