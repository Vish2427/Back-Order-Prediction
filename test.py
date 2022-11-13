from order.pipeline.training_pipeline import TrainingPipeline

def run():
        config = TrainingPipeline()
        return config.run_pipeline()

if __name__ =="__main__":
        run()