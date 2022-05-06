import os
import time
from pathlib import Path

import numpy as np
import pytest
from jina import Document, DocumentArray, Executor

from executor import PostgreSQLReader

cur_dir = os.path.dirname(os.path.abspath(__file__))
compose_yml = os.path.abspath(os.path.join(cur_dir, 'docker-compose.yml'))


d_embedding = np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
c_embedding = np.array([2, 2, 2, 2, 2, 2, 2], dtype=np.float32)


def get_documents(chunks, same_content, nr=10, index_start=0, same_tag_content=None):
    next_chunk_id = nr + index_start
    for i in range(index_start, nr + index_start):
        d = Document()
        d.id = i
        if same_content:
            d.text = 'hello world'
            d.embedding = np.random.random(d_embedding.shape)
        else:
            d.text = f'hello world {i}'
            d.embedding = np.random.random(d_embedding.shape)
        if same_tag_content:
            d.tags['field'] = 'tag data'
        elif same_tag_content is False:
            d.tags['field'] = f'tag data {i}'
        for j in range(chunks):
            c = Document()
            c.id = next_chunk_id
            if same_content:
                c.text = 'hello world from chunk'
                c.embedding = np.random.random(c_embedding.shape)
            else:
                c.text = f'hello world from chunk {j}'
                c.embedding = np.random.random(c_embedding.shape)
            if same_tag_content:
                c.tags['field'] = 'tag data'
            elif same_tag_content is False:
                c.tags['field'] = f'tag data {next_chunk_id}'
            next_chunk_id += 1
            d.chunks.append(c)
        yield d


@pytest.fixture()
def docker_compose(request):
    os.system(
        f'docker-compose -f {request.param} --project-directory . up  --build -d '
        f'--remove-orphans'
    )
    time.sleep(5)
    yield
    os.system(
        f'docker-compose -f {request.param} --project-directory . down '
        f'--remove-orphans'
    )


def test_config():
    ex = Executor.load_config(
        str(Path(__file__).parents[1] / 'config.yml'), uses_with={'dry_run': True}
    )
    assert ex.username == 'postgres'


@pytest.mark.parametrize('docker_compose', [compose_yml], indirect=['docker_compose'])
def test_postgres(docker_compose):
    reader = PostgreSQLReader()

    emb = np.random.random(10).astype(np.float32)

    doc = Document(embedding=emb)
    da = DocumentArray([doc])

    query1 = DocumentArray([Document(id=doc.id)])

    reader.add(da, parameters={})
    reader.search(query1, parameters={})

    np.testing.assert_array_equal(query1[0].embedding, emb)

    query2 = DocumentArray([Document(id=doc.id)])
    reader.search(query2, parameters={'return_embeddings': False})
    assert query2[0].embedding is None

    query3 = DocumentArray([Document(id='000')])
    reader.search(query3, parameters={'return_embeddings': False})
    assert query3[0].content is None
    assert query3[0].id == '000'
