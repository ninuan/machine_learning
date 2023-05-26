import json
import os

# 用户类
class User:
    def __init__(self, username, password, gender='', age=0, address=''):
        self.username = username
        self.password = password
        self.gender = gender
        self.age = age
        self.address = address

    def to_dict(self):
        return {
            'username': self.username,
            'password': self.password,
            'gender': self.gender,
            'age': self.age,
            'address': self.address
        }

# 用户系统类
class UserSystem:
    def __init__(self):
        self.users = []
        self.current_user = None
        self.load_users()

    def load_users(self):
        if os.path.exists('users.json'):
            with open('users.json', 'r') as f:
                users_data = json.load(f)
                for user_data in users_data:
                    user = User(
                        user_data['username'],
                        user_data['password'],
                        user_data['gender'],
                        user_data['age'],
                        user_data['address']
                    )
                    self.users.append(user)

    def save_users(self):
        users_data = [user.to_dict() for user in self.users]
        with open('users.json', 'w') as f:
            json.dump(users_data, f, indent=4)

    def register(self, username, password, gender='', age=0, address=''):
        for user in self.users:
            if user.username == username:
                print('用户名已存在！请使用其他用户名进行注册！')
                return
        user = User(username, password, gender, age, address)
        self.users.append(user)
        self.current_user = user
        self.save_users()
        print('注册成功！')

    def login(self, username, password):
        for user in self.users:
            if user.username == username and user.password == password:
                self.current_user = user
                print('恭喜您，登录成功！')
                print('进入二级文章页面')
                return
        print('用户名或密码错误！')

    def view(self):
        for user in self.users:
            print(f'用户名:{user.username},密码:{user.password}')

    def logout(self):
        self.current_user = None

    def create_article(self, title, content):
        filename = title.lower().replace(' ', '_') + '.txt'
        if os.path.exists(filename):
            print('文章名已存在，请选择其他标题！')
            return
        with open(filename, 'w') as f:
            f.write(content)
        print('文章创建成功！')

    def read_article(self, title):
        filename = title.lower().replace(' ', '_') + '.txt'
        try:
            with open(filename, 'r') as f:
                content = f.read()
            print(f"标题: {title}")
            print(f"内容: {content}")
        except FileNotFoundError:
            print('找不到该文章！')

    def update_article(self, title, new_content):
        filename = title.lower().replace(' ', '_') + '.txt'
        if not os.path.exists(filename):
            print('该文章不存在！')
            return
        try:
            with open(filename, 'w') as f:
                f.write(new_content)
            print('文章修改成功！')
        except FileNotFoundError:
            print('找不到该文章！')

    def delete_article(self, title):
        filename = title.lower().replace(' ', '_') + '.txt'
        try:
            os.remove(filename)
            print('文章删除成功！')
        except FileNotFoundError:
            print('找不到该文章！')

# 控制台交互
user_system = UserSystem()

while True:
    if user_system.current_user is None:
        print('-----------欢迎来到网站管理系统--------------')
        print('1. 注册')
        print('2. 登录')
        print('3. 退出')
        print('-----------------------------------------')
        choice = input('请输入选项：')

        if choice == '1':
            username = input('请输入用户名：')
            password = input('请输入密码：')
            gender = input('请输入性别（可选）：')
            age = input('请输入年龄（可选）：')
            address = input('请输入地址（可选）：')
            user_system.register(username, password, gender, age, address)

        elif choice == '2':
            user_system.view()
            username = input('请输入用户名：')
            password = input('请输入密码：')
            user_system.login(username, password)

        elif choice == '3':
            print('已退出系统！')
            break

        else:
            print('无效选项！')
    else:
        print('-----------------------------------------')
        print('1. 写文章')
        print('2. 查看文章')
        print('3. 修改文章')
        print('4. 删除文章')
        print('5. 退出登录')
        print('-----------------------------------------')

        choice = input('请输入选项：')

        if choice == '1':
            title = input('请输入文章标题：')
            content = input('请输入文章内容：')
            user_system.create_article(title, content)

        elif choice == '2':
            title = input('请输入要查看的文章标题：')
            user_system.read_article(title)

        elif choice == '3':
            title = input('请输入要修改的文章标题：')
            new_content = input('请输入新的文章内容：')
            user_system.update_article(title, new_content)

        elif choice == '4':
            title = input('请输入要删除的文章标题：')
            user_system.delete_article(title)

        elif choice == '5':
            user_system.logout()
            print('已退出登录！')

        else:
            print('无效选项！')