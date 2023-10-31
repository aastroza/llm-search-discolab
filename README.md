# llm-search-discolab
Discolab's Generative Search

## Installation

```
conda create --name llm-search-discolab -c conda-forge python=3.10
conda activate llm-search-discolab
pip install -r requirements.txt
```

## Example

```
curl --location 'https://aastroza--new-rag-discolab-api.modal.run/stream' \
--header 'Content-Type: application/json' \
--data '{
    "query": "El banco central",
    "document1_id": 1,
    "document2_id": 2
}' \
--no-buffer
```