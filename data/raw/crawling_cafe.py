import time
from selenium import webdriver
import csv
import pandas as pd
from bs4 import BeautifulSoup as bs
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from datetime import datetime
import re

# --- 설정 구간 ---
url = 'https://nid.naver.com/nidlogin.login'
id = 'id'
pw = 'password'
baseurl = 'https://cafe.naver.com/makemywedding'
clubid = '28757979'
menuid = '14'
start_page = 0
end_page = 9
# ----------------

chrome_options = Options()
chrome_options.add_experimental_option("detach", True)
chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])
browser = webdriver.Chrome(options=chrome_options)

# 1. 로그인 로직
browser.get(url)
browser.implicitly_wait(2)
browser.execute_script(f"document.getElementsByName('id')[0].value='{id}'")
browser.execute_script(f"document.getElementsByName('pw')[0].value='{pw}'")
browser.find_element(By.XPATH, '//*[@id="log.login"]').click()
time.sleep(2) # 로그인 후 안정화 대기

cafemenuurl = f'{baseurl}/ArticleList.nhn?search.clubid={clubid}&search.menuid={menuid}&search.boardtype=L&userDisplay=50'

i = start_page
while i < end_page: # 1. 무한루프 방지: 정해진 페이지수만큼만 실행
    pageNum = i + 1
    print(f"\n현재 {pageNum}페이지 크롤링 중...")
    
    browser.get(f'{cafemenuurl}&search.page={str(pageNum)}')
    time.sleep(2)
    browser.switch_to.frame('cafe_main')

    soup = bs(browser.page_source, 'html.parser')
    
    # 2. 게시글 목록 영역만 정확히 타겟팅
    # 공지사항 영역과 일반 게시글 영역이 나뉘어 있으므로 구분 필요
    datas = soup.select("#main-area > div.article-board > table > tbody > tr")

    if not datas:
        print("더 이상 게시글이 없습니다. 종료합니다.")
        break

    for data in datas:
        try:
            # 3. 공지사항 제외 (번호가 없는 글 스킵)
            num_element = data.select_one('.td_article .inner_number')
            if not num_element: continue 
            
            article_info = data.select(".article")
            if not article_info: continue

            article_href = article_info[0].attrs['href']
            # 주소 형식이 이미 풀 주소인 경우와 아닌 경우 대비
            if "https" not in article_href:
                article_full_url = f'https://cafe.naver.com{article_href}'
            else:
                article_full_url = article_href

            # 게시글 상세 페이지 이동
            browser.get(article_full_url)
            time.sleep(1.5) # 로딩 대기
            browser.switch_to.frame('cafe_main')
            
            article_soup = bs(browser.page_source, 'html.parser')
            content_box = article_soup.find('div', class_='ArticleContentBox')

            if not content_box: continue

            # 데이터 추출
            title = content_box.find("h3", {"class" : "title_text"})
            title = title.text.strip() if title else "제목없음"
            
            date_el = content_box.find("span", {"class": "date"})
            date_val = date_el.text.strip() if date_el else "null"
            
            content_el = content_box.find("div", {"class": "se-main-container"})
            if content_el:
                content = content_el.get_text(separator=' ', strip=True)
                content = " ".join(content.split()) # 줄바꿈/공백 정리
            else:
                content = "내용없음(복사금지 혹은 이미지글)"

            author = content_box.find("button", {"class": "nickname"})
            author = author.text.strip() if author else "null"

            # 4. 파일 저장 (매 글마다 저장)
            with open('crawling_2026_0105.csv', 'a+', newline='', encoding="utf-8-sig") as f:
                wr = csv.writer(f)
                wr.writerow([title, author, date_val, article_full_url, content])

            # 다시 목록으로 돌아가기 (browser.back() 대신 리스트 재접속이 안전)
            browser.get(f'{cafemenuurl}&search.page={str(pageNum)}')
            browser.switch_to.frame('cafe_main')
            time.sleep(1)

        except Exception as e:
            print(f"글 처리 중 에러 발생(스킵): {e}")
            continue

    i += 1 # 5. 핵심: 다음 페이지로 넘어가기 위한 증감식 추가
    time.sleep(1)

print("\n크롤링 완료!")
browser.quit()