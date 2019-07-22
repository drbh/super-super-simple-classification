wget -O model "https://tfhub.dev/google/universal-sentence-encoder-large/3?tf-hub-format=compressed" 
mkdir universal-sentence-encoder-large
tar -xvzf model -C ./universal-sentence-encoder-large
rm model