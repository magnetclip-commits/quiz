--현재 세팅되어 있는 교수 프롬프트 값 확인
SELECT * --profile_body 값 확인 
FROM aita_profile_mst apm 
WHERE user_id = '99999'
AND profile_type = 'USR'
;

--과거 교수 프롬프트 값 확인 
SELECT * --profile_body 값 확인 
FROM aita_profile_hist aph  
WHERE user_id = '99999'
AND profile_type = 'USR'
ORDER BY hist_id DESC --내림차순 
;

--교수 프롬프트 사용여부 변경시 
UPDATE aita_profile_mst
SET use_yn = 'N' -- OR 'Y'
WHERE user_id = '99999'
AND profile_type = 'USR'
;

--과목 프롬프트 사용여부 변경시 
UPDATE aita_profile_mst
SET use_yn = 'N' -- OR 'Y'
WHERE user_id = '99999'
AND cls_id = '2026-10-99999-2-01'
AND profile_type = 'CLS'
;
