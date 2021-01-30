

gcloud ai-platform jobs submit training mnist_`date +"%s"` \
--python-version 3.7 \
--runtime-version 2.3 \
--scale-tier BASIC_GPU \
--package-path ./trainer \
--module-name trainer.task \
--region europe-west1 \
--job-dir gs://alepadel-20200122-kschool/tmp \
--config ./bin/hyper.yaml \
-- \
--model-output-path gs://alepadel-20200122-kschool/models \
--hypertune
