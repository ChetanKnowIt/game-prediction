
# game-prediction

A project repository for game prediction focusing on cricket matches mainly T20 IPL




## Roadmap

- Add type hints using [typing module](https://docs.python.org/3.9/library/typing.html)
- Add required preprocessing in ```init```method [here](https://github.com/ChetanKnowIt/game-prediction/blob/main/mymodelfile.py#L5)
- Add required output format in ```predict``` method [here](https://github.com/ChetanKnowIt/game-prediction/blob/main/mymodelfile.py#L22)
- Evaluate and test model to keep mean_absolute_error minimum




## Usage with Docker for Linux/Ubuntu:

```Shell
docker run \
--mount type=bind,src=$(pwd)/test_file.csv,dst=/var/test_file.csv \
--mount type=bind,src=$(pwd)/submission.csv,dst=/var/submission.csv \
--mount type=bind,src=$(pwd)/mymodelfile.py,dst=/var/mymodelfile.py \
--mount type=bind,src=$(pwd)/logs.txt,dst=/var/logs.txt \
swarnimsoni/iitmbsiplcontest2023:latest
```


## License

[MIT](https://choosealicense.com/licenses/mit/)


## Authors

- [@ChetanKnowit](https://github.com/ChetanKnowIt)

