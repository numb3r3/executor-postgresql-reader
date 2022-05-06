__copyright__ = 'Copyright (c) 2021 Jina AI Limited. All rights reserved.'
__license__ = 'Apache-2.0'


from typing import Dict, Optional

import numpy as np
from jina import Document, DocumentArray, Executor, requests

from .postgreshandler import PostgreSQLHandler


class PostgreSQLReader(Executor):
    """:class:`PostgreSQLReader` PostgreSQL-based Document Reader."""

    def __init__(
        self,
        hostname: str = '127.0.0.1',
        port: int = 5432,
        username: str = 'postgres',
        password: str = '123456',
        database: str = 'postgres',
        table: str = 'default_table',
        max_connections=5,
        traversal_paths: str = '@r',
        return_embeddings: bool = True,
        dry_run: bool = False,
        dump_dtype: type = np.float64,
        *args,
        **kwargs,
    ):
        """
        Initialize the PostgreSQLStorage.

        :param hostname: hostname of the machine
        :param port: the port
        :param username: the username to authenticate
        :param password: the password to authenticate
        :param database: the database name
        :param table: the table name to use
        :param return_embeddings: whether to return embeddings on search or
        not
        :param dry_run: If True, no database connection will be build.
        """
        super().__init__(*args, **kwargs)

        self.default_traversal_paths = traversal_paths
        self.hostname = hostname
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.table = table

        self.handler = PostgreSQLHandler(
            hostname=self.hostname,
            port=self.port,
            username=self.username,
            password=self.password,
            database=self.database,
            table=self.table,
            max_connections=max_connections,
            dry_run=dry_run,
            dump_dtype=dump_dtype,
        )
        self.default_return_embeddings = return_embeddings

    @property
    def dump_dtype(self):
        return self.handler.dump_dtype

    @property
    def size(self):
        """Obtain the size of the table

        .. # noqa: DAR201
        """
        return self.handler.get_size()

    def close(self) -> None:
        """
        Close the connections in the connection pool
        """
        self.handler.close()

    @requests(on='/search')
    def search(self, docs: DocumentArray, parameters: Dict, **kwargs):
        """Get the Documents by the ids of the docs in the DocArray

        :param docs: the DocumentArray to search
         with (they only need to have the `.id` set)
        :param parameters: the parameters to this request
        """
        if docs is None:
            return
        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )

        self.handler.search(
            docs[traversal_paths],
            return_embeddings=parameters.get(
                'return_embeddings', self.default_return_embeddings
            ),
        )

    def add(
        self, docs: Optional[DocumentArray] = None, parameters: Dict = {}, **kwargs
    ):
        """Add Documents to Postgres
        :param docs: list of Documents
        :param parameters: parameters to the request
        """
        if docs is None:
            return
        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )
        with self.handler as postgres_handler:
            postgres_handler.add(docs[traversal_paths])

    @property
    def initialized(self, **kwargs):
        """
        Whether the PSQL connection has been initialized
        """
        return hasattr(self.handler, 'postgreSQL_pool')
