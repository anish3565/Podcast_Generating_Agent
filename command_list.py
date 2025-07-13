Bash
1. docker build -t blog-summarizer-img .
2. docker run -it --env-file .env blog-summarizer-img
3. docker build -t final-podcast-img .
4. docker run -it --env-file .env final-podcast-img