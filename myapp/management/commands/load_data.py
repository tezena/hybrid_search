import pandas as pd
from django.core.management.base import BaseCommand

import os
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from myapp.qdrant_helpers import upload_data_to_qdrant, create_database

class Command(BaseCommand):
    help = 'Load data from CSV '

    def handle(self, *args, **kwargs):


        try:
            # create_database()
            response=upload_data_to_qdrant()
            
            if response is not None:
              print('Successfully loaded data and initialized vectorization')
            else:
              print('Somthing wrong!!!')



        except Exception as e:
            raise Exception(f"Failed to load data: {e}")
