FROM daguan:1

ADD . /data
WORKDIR /data
# RUN groupadd -g 1000 ps && \
#     useradd -u 1000 ps -g ps 
# RUN chmod -R o+rwx /home/
# USER 1000

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# CMD ["sh", "run.sh"]