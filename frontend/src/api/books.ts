import http from "./http";

export interface BookListItem {
  id: string;
  title: string;
  description: string | null;
  cover_image_url: string | null;
  target_age_min: number;
  target_age_max: number;
  categories: string[];
  author: string | null;
  created_at: string;
}

export interface BookSection {
  id: string;
  order: number;
  title: string | null;
  text: string;
}

export interface BookDetail extends BookListItem {
  sections: BookSection[];
}

export interface BookListResponse {
  total: number;
  items: BookListItem[];
}

export async function fetchBooks(params: {
  keyword?: string;
  categories?: string[];
}): Promise<BookListResponse> {
  const query = new URLSearchParams();
  if (params.keyword) query.set("keyword", params.keyword);
  params.categories?.forEach((c) => query.append("categories", c));

  const { data } = await http.get<BookListResponse>(`/books?${query}`);
  return data;
}

export async function fetchCategories(): Promise<string[]> {
  const { data } = await http.get<string[]>("/books/categories");
  return data;
}

export async function fetchBookDetail(id: string): Promise<BookDetail> {
  const { data } = await http.get<BookDetail>(`/books/${id}`);
  return data;
}
