FROM python:3.10

# Use local repo
RUN mkdir -p /usr/src/app
COPY . /usr/src/app
WORKDIR /usr/src/app

RUN python3 -m pip install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt


EXPOSE 7070
ENTRYPOINT ["python"]
CMD ["src/app.py"]
