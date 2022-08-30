# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
# pip3 install tornado

import tornado.ioloop
import tornado.web
import json

from aixcoder.aixcode import AIXCode

AIXCode1 = AIXCode('codegen-350M-multi')
AIXCode2 = AIXCode('codegen-2B-multi')


def get_body_json(body):
    body_decode = body.decode()
    body_json = json.loads(body_decode)
    return body_json


class PingHandler(tornado.web.RequestHandler):
    def get(self):
        print(f'request:{self.request.full_url()}')
        self.write("Pong!")

    def post(self):
        body_json = get_body_json(self.request.body)
        print(f'request:{body_json}')
        self.write("Pong!")


class AIX1Handler(tornado.web.RequestHandler):
    def get(self):
        """get请求"""
        print(f'request:{self.request.full_url()}')
        x = self.get_argument('x')
        self.write(AIXCode1.aixcode(x))

    def post(self):
        '''post请求'''
        body_json = get_body_json(self.request.body)
        print(f'request:{body_json}')

        x = body_json.get("x")
        self.write(AIXCode1.aixcode(x))


class AIX2Handler(tornado.web.RequestHandler):
    def get(self):
        """get请求"""
        print(f'request:{self.request.full_url()}')
        x = self.get_argument('x')
        self.write(AIXCode2.aixcode(x))

    def post(self):
        '''post请求'''
        body_json = get_body_json(self.request.body)
        print(f'request:{body_json}')
        x = body_json.get("x")
        self.write(AIXCode2.aixcode(x))


if __name__ == "__main__":
    # 注册路由
    app = tornado.web.Application([
        (r"/ping", PingHandler),
        (r"/aix1", AIX1Handler),
        (r"/aix2", AIX2Handler),
    ])

    # 监听端口
    port = 8888
    app.listen(port)
    print(f'AIXCoder Started, Listening on Port:{port}')
    # 启动应用程序
    tornado.ioloop.IOLoop.instance().start()
