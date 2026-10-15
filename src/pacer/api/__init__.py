"""HTTP API (FastAPI): the entry point for the web interface and other clients.

May import from: services, domain, ai (the pipelines and their errors), and repositories
(only in dependencies.py, to build the services and pipelines for each request).

    app.py           the application, routers and error handlers wired together
    dependencies.py  Dependency Injection: session, actor, repositories, services
    schemas.py       request bodies
    errors.py        service and domain errors -> HTTP status codes
    routers/         the endpoints, one file per resource
"""
