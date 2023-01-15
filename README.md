# Information Retrieval Project

This project aims to implement and evaluate various information retrieval techniques for the entire Wikipedia corpus dataset. The project is developed as a part of the Information Retrieval course at Ben-Gurion University.

## Explanations on the developing process and the different notebooks

At the beggining of the project, we created and developed the "preperations" notebook. This notebook includes all the stages of creating the data and objects required for the first small engine prototype. we created small corpus of about 1000 documents, writing it as a csv file to the local machine and then reading it in order to create a rdd from it from running to running.
In the "first_prototype_colab", we implemented all the information retrieval techniques we learnt in class in order to try our first prototype. We did all that with lot of forward thinking - you can note that all the reading and calculations were made with Pyspark's rdd's in order to make things faster, which later on did not come to fruition.
After we got results from the prototype, we managed to keep forward to the "engine_gcp" notebook that was written and running on GCP cloud on the full Wikipedia corpus.
After we managed to create the full index for the first time, we wrote it to the bucket of the cloud, and from there onwards we created a small scripts notebook - "import_bins_from_bucket_to_gcp". In this notebook we read the 'bin' and 'pkl' files from the bucket to the RAM of the cluster/machine so we don't have to create a huge index every time.
"engine_gcp" is a big notebook which includes both the development stages and the model building and optimization stages. The notebook is structured chronologically and arranged according to titles and documentations.

### Prerequisites

- Python 3.6 or later
- Required libraries: `pyspark`, `collections`, `itertools`, `re`, `pandas`, `nltk`, `pathlib`, `pickle`, `storage`, `storage`, `math`

## Authors

- Yuval Schwartz
- Amit Kravchik

## License

This project is licensed under the Ben-Gurion License.

## Contact

- For any questions or issues, please open an issue on the [GitHub page](https://github.com/YuvalSchwartz/IR-Engine/issues) or contact the authors directly.
