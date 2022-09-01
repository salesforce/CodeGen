# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
# pip3 install tornado

import tornado.ioloop
import tornado.web
import tornado.gen
from concurrent.futures import ThreadPoolExecutor
from tornado.concurrent import run_on_executor
import json

from aixcoder.aixcode import AIXCode

AIXCode1 = AIXCode('codegen-350M-multi')
# AIXCode2 = AIXCode('codegen-2B-multi')
AIXCode3 = AIXCode('codegen-350M-mono')


def get_body_json(body):
    body_decode = body.decode()
    body_json = json.loads(body_decode)
    return body_json


class PingHandler(tornado.web.RequestHandler):
    # 跨域配置：https://github.com/tornadoweb/tornado/issues/2104
    def initialize(self):
        self.set_default_header()

    def set_default_header(self):
        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Access-Control-Allow-Headers', '*')
        self.set_header('Access-Control-Max-Age', 1000)
        self.set_header('Content-type', 'application/json')
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.set_header('Access-Control-Allow-Headers',
                        'Content-Type, Access-Control-Allow-Origin, Access-Control-Allow-Headers, X-Requested-By, Access-Control-Allow-Methods')

    @tornado.gen.coroutine
    def get(self):
        print(f'request:{self.request.full_url()}')
        self.write("Pong!")

    @tornado.gen.coroutine
    def post(self):
        print(f'request:{self.request.full_url()}')
        body_json = get_body_json(self.request.body)
        print(f'request:{body_json}')
        self.write("Pong!")


class AIX1Handler(tornado.web.RequestHandler):
    # 跨域配置：https://github.com/tornadoweb/tornado/issues/2104
    def initialize(self):
        self.set_default_header()

    def set_default_header(self):
        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Access-Control-Allow-Headers', '*')
        self.set_header('Access-Control-Max-Age', 1000)
        self.set_header('Content-type', 'application/json')
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.set_header('Access-Control-Allow-Headers',
                        'Content-Type, Access-Control-Allow-Origin, Access-Control-Allow-Headers, X-Requested-By, Access-Control-Allow-Methods')

    executor = ThreadPoolExecutor(32)

    @run_on_executor
    def aixcode(self, x):
        return AIXCode1.aixcode(x)

    @tornado.gen.coroutine
    def get(self):
        """get请求"""
        print(f'request:{self.request.full_url()}')
        x = self.get_argument('x')
        y = yield self.aixcode(x)
        print(y)
        self.write(y)

    @tornado.gen.coroutine
    def post(self):
        '''post请求'''
        print(f'request:{self.request.full_url()}')
        body_json = get_body_json(self.request.body)
        print(f'request:{body_json}')
        x = body_json.get("x")
        y = yield self.aixcode(x)
        print(y)
        self.write(y)


# class AIX2Handler(tornado.web.RequestHandler):
#     # 跨域配置：https://github.com/tornadoweb/tornado/issues/2104
#     def initialize(self):
#         self.set_default_header()
#
#     def set_default_header(self):
#         self.set_header('Access-Control-Allow-Origin', '*')
#         self.set_header('Access-Control-Allow-Headers', '*')
#         self.set_header('Access-Control-Max-Age', 1000)
#         self.set_header('Content-type', 'application/json')
#         self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
#         self.set_header('Access-Control-Allow-Headers',
#                         'Content-Type, Access-Control-Allow-Origin, Access-Control-Allow-Headers, X-Requested-By, Access-Control-Allow-Methods')
#
#     executor = ThreadPoolExecutor(32)
#
#     @run_on_executor
#     def aixcode(self, x):
#         return AIXCode2.aixcode(x)
#
#     @tornado.gen.coroutine
#     def get(self):
#         """get请求"""
#         print(f'request:{self.request.full_url()}')
#         x = self.get_argument('x')
#         y = yield self.aixcode(x)
#         print(y)
#         self.write(y)
#
#     @tornado.gen.coroutine
#     def post(self):
#         '''post请求'''
#         print(f'request:{self.request.full_url()}')
#         body_json = get_body_json(self.request.body)
#         print(f'request:{body_json}')
#         x = body_json.get("x")
#         y = yield self.aixcode(x)
#         print(y)
#         self.write(y)


class AIX3Handler(tornado.web.RequestHandler):
    # 跨域配置：https://github.com/tornadoweb/tornado/issues/2104
    def initialize(self):
        self.set_default_header()

    def set_default_header(self):
        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Access-Control-Allow-Headers', '*')
        self.set_header('Access-Control-Max-Age', 1000)
        self.set_header('Content-type', 'application/json')
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.set_header('Access-Control-Allow-Headers',
                        'Content-Type, Access-Control-Allow-Origin, Access-Control-Allow-Headers, X-Requested-By, Access-Control-Allow-Methods')

    executor = ThreadPoolExecutor(32)

    @run_on_executor
    def aixcode(self, x):
        return AIXCode3.aixcode(x)

    @tornado.gen.coroutine
    def get(self):
        """get请求"""
        print(f'request:{self.request.full_url()}')
        x = self.get_argument('x')
        y = yield self.aixcode(x)
        print(y)
        self.write(y)

    @tornado.gen.coroutine
    def post(self):
        '''post请求'''
        print(f'request:{self.request.full_url()}')
        body_json = get_body_json(self.request.body)
        print(f'request:{body_json}')
        x = body_json.get("x")
        y = yield self.aixcode(x)
        print(y)
        self.write(y)


if __name__ == "__main__":
    # 注册路由
    app = tornado.web.Application([
        (r"/ping", PingHandler),
        (r"/aix1", AIX1Handler),
        # (r"/aix2", AIX2Handler),
        (r"/aix3", AIX3Handler),
    ])

    # 监听端口
    port = 8888
    app.listen(port)
    print(f'AIXCoder Started, Listening on Port:{port}')
    # 启动应用程序
    tornado.ioloop.IOLoop.instance().start()
