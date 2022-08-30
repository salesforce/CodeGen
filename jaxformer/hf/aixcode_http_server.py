# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
# pip3 install tornado

import tornado.ioloop
import tornado.web
import json

from jaxformer.hf.aixcode import AIXCode

AIXCode1 = AIXCode('codegen-350M-multi')
AIXCode2 = AIXCode('codegen-2B-multi')


class PingHandler(tornado.web.RequestHandler):
    def get(self):
        print(f'request:{json.dumps(self.request.body)}')
        self.write("Pong!")

    def post(self):
        print(f'request:{json.dumps(self.request.body)}')
        self.write("Pong!")


class AIX1Handler(tornado.web.RequestHandler):
    def get(self):
        """get请求"""
        print(f'request:{json.dumps(self.request.body)}')
        input = self.get_argument('input')
        self.write(AIXCode1.aixcode(input))

    def post(self):
        '''post请求'''
        print(f'request:{json.dumps(self.request.body)}')
        body = self.request.body
        body_decode = body.decode()
        body_json = json.loads(body_decode)
        input = body_json.get("input")
        self.write(AIXCode1.aixcode(input))


class AIX2Handler(tornado.web.RequestHandler):
    def get(self):
        """get请求"""
        print(f'request:{json.dumps(self.request.body)}')
        input = self.get_argument('input')
        self.write(AIXCode2.aixcode(input))

    def post(self):
        '''post请求'''
        print(f'request:{json.dumps(self.request.body)}')
        body = self.request.body
        body_decode = body.decode()
        body_json = json.loads(body_decode)
        input = body_json.get("input")
        self.write(AIXCode2.aixcode(input))


if __name__ == "__main__":
    # 注册路由
    app = tornado.web.Application([
        (r"/ping", PingHandler),
        (r"/aix1", AIX1Handler),
        (r"/aix2", AIX2Handler),
    ])

    # 监听端口
    port = 8868
    app.listen(port)
    print(f'AIXCoder Started, Listening on Port:{port}')
    # 启动应用程序
    tornado.ioloop.IOLoop.instance().start()
