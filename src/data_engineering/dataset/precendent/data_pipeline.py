class Pipelines:
    def pipeline_0():
        return
    
    def pipeline_1():
        return
    
    
def install(data, pipeline:str = "pipeline_0", kwargs:dict = {}):
    return getattr(Pipelines, pipeline)(data, **kwargs)