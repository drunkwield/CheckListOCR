from flask import Flask, request, render_template
import os
import cv2
import numpy as np
import pytesseract
import copy

# ===================== 核心类 ======================

class question:
    def __init__(self, question, answer, RightAnswer):
        self.question = question
        self.Answer = answer
        self.RightAnswer = RightAnswer
        self.isRight = self.Answer == self.RightAnswer

class table:
    def __init__(self, mat=[], answer={}):
        self.mat = mat
        self.RowNumber = len(self.mat) - 1
        self.ColNumber = len(self.mat[0]) - 1
        self.ColTitle = self.mat[0]
        self.RowTitle = [self.mat[i][0] for i in range(self.ColNumber)]
        self.answer = answer
        self.score = 0
        self.mark = []

    def setColTitle(self, ColTitle):
        self.ColTitle = ColTitle
        for i in range(self.ColNumber):
            self.mat[0][i+1] = ColTitle[i]

    def setRowTitle(self, RowTitle):
        self.RowTitle = RowTitle
        for i in range(self.RowNumber):
            self.mat[i+1][0] = RowTitle[i]

    def getCell(self, row, col):
        return self.mat[row][col]

    def setAnswer(self, answer):
        if len(answer) == self.RowNumber:
            self.answer = copy.copy(answer)
        else:
            raise ValueError('answer length error or type error')

    def checkAnswer(self):
        self.score = 0
        self.mark = []
        for i in range(1, self.RowNumber + 1):
            ans = ""
            for k in range(1, self.ColNumber + 1):
                if self.getCell(i, k) == '√':
                    ans = self.ColTitle[k - 1]
            right = self.answer[i - 1]
            tmp = question(i, ans, right)
            self.mark.append(copy.copy(tmp))
            if tmp.isRight:
                self.score += 1
        return self.score

    def getResultTable(self):
        result = []
        for m in self.mark:
            result.append([m.question, m.Answer, m.RightAnswer, '✔' if m.isRight else '✘'])
        result.append(['Total Score', self.score, '', ''])
        return result
    
    


class mesh:
    def __init__(self, row, col, image_path):
        self.mat = [[] for _ in range(row)]
        self.cells = self.detect_table_cells(image_path)
        j = row - 1
        for i in range(len(self.cells)):
            if len(self.mat[j]) == col:
                j -= 1
            self.mat[j].insert(0, self.cells[i])

    def detect_table_cells(self, image_path):
        def order_points(pts):
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            return rect

        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        grid = cv2.add(horizontal, vertical)
        contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cells = []
        for cnt in contours:
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) == 4:
                rect = order_points(approx.reshape(-1, 2))
                cells.append(rect)
        return cells[1:]  # 去掉第一个最外框

class reconginser:
    def __init__(self, row, col, path):
        self.path = path
        self.row = row
        self.col = col
        self.img = cv2.imread(self.path)
        self.mesh = mesh(row, col, path)
        self.res = []

    def cell2xywh(self, cell):
        x = int(max(cell[0][0], cell[3][0]))
        w = int(min(cell[1][0], cell[2][0]) - x)
        y = int(max(cell[0][1], cell[1][1]))
        h = int(min(cell[2][1], cell[3][1]) - y)
        return x+5, y+2, w-5, h-2

    def ocr_specific_area(self, x, y, w, h):
        roi = self.img[y:y + h, x:x + w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        text = pytesseract.image_to_string(thresh, config=r'--oem 3 --psm 6')
        return text.strip()

    def pic_processing(self):
        for i in range(self.row):
            tmp = []
            for j in range(self.col):
                x, y, w, h = self.cell2xywh(self.mesh.mat[i][j])
                txt = self.ocr_specific_area(x, y, w, h)
                tmp.append(txt)
            self.res.append(tmp)

    def res2table(self):
        mat = copy.deepcopy(self.res)
        for i in range(1, len(mat)):
            for j in range(1, len(mat[0])):
                if mat[i][j]:
                    mat[i][j] = '√'
        tb = table(mat)
        return tb

# ===================== Flask 应用 ======================

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET'])
def index():
    return render_template('upload.html')

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files['image']
    correct_str = request.form.get('answers', '').strip().upper() 

    # 校验24个字母
    if not file or len(correct_str) != 24 or not all(c in 'ABCDE' for c in correct_str):
        return "Invalid input. Please make sure the answers are 24 letters (A-E).", 400

    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    rec = reconginser(25, 6, filepath)
    rec.pic_processing()
    tb = rec.res2table()
    tb.setColTitle(['A', 'B', 'C', 'D', 'E'])
    tb.setRowTitle([i for i in range(1, len(tb.mat) )])

    # correct_list 长度24，应用到第2~25题
    # 题号1是标题，不验证，故给第1题一个空或占位符
    correct_list = list(correct_str)  # 长度24，跟RowNumber一致
    tb.setAnswer(correct_list)
    tb.checkAnswer()


    result = [[q.question, q.Answer, q.RightAnswer, '✔' if q.isRight else '✘'] for q in tb.mark]
    result.append(['Total Score', tb.checkAnswer(), '', ''])
    return render_template("upload.html", result=result)

####################################################################

if __name__ == '__main__':
    app.run(port=8848, debug=True)
 