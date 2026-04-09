import http from "./http";

export interface QuestionResponse {
  question: string;
}

export interface EvaluateResponse {
  score: number;          // 0.0–1.0
  feedback: string;
  correct_hint: string;
}

export async function generateQuestion(sectionText: string): Promise<QuestionResponse> {
  const { data } = await http.post<QuestionResponse>("/qa/question", {
    section_text: sectionText,
  });
  return data;
}

export async function evaluateAnswer(
  question: string,
  context: string,
  userAnswer: string
): Promise<EvaluateResponse> {
  const { data } = await http.post<EvaluateResponse>("/qa/evaluate", {
    question,
    context,
    user_answer: userAnswer,
  });
  return data;
}

export async function generateFollowup(
  context: string,
  prevQuestion: string,
  userAnswer: string,
  feedback: string
): Promise<QuestionResponse> {
  const { data } = await http.post<QuestionResponse>("/qa/followup", {
    context,
    prev_question: prevQuestion,
    user_answer: userAnswer,
    feedback,
  });
  return data;
}
