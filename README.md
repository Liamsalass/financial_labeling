## Preprocessing
Donwload the finer dataset:
```
import datasets

finer = datasets.load_dataset("nlpaueb/finer-139")
```

After running the above code, it will tell you in the output where the cached dataset has been saved to.

Change this part to your location of the cached dataset in `fix_data.py` and run:
![image](https://github.com/Liamsalass/financial_labeling/assets/125309468/693917c6-2920-4eca-aa69-99bdc75ba916)

Now run `run_finer.sh`
