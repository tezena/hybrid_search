from django.shortcuts import render


import json
from django.http import JsonResponse
from .qdrant_helpers import search_query
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from rest_framework.response import Response

@api_view(['POST'])
def search_view(request):
    try:
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        # content = 
        # data=json.load(request.body)
        query_text=body['query_text']
        query_keyword=body['query_keyword']

        if not query_text or not query_keyword:
            return Response({'error': 'Invalid JSON'}, status=400)

        results = search_query(query_text, query_keyword)
        
        if results is not None:
            return Response({'results': results}, status=200)
        else:
            return Response({'error': 'No results found'}, status=404)
            
    except json.JSONDecodeError :
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
