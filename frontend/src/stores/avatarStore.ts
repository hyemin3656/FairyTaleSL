import { create } from "zustand";
import { persist } from "zustand/middleware";

export interface AvatarOption {
  id: string;
  name: string;
  url: string;       // VRM/GLB 파일 경로
  preview: string;   // 썸네일 이미지 경로
  description: string;
}

export const AVATAR_OPTIONS: AvatarOption[] = [
  {
    id: "default",
    name: "성준",
    url: "/avatar.glb",
    preview: "/avatars/preview_default.png",
    description: "씩씩하고 용감한 남자 아바타",
  },
  {
    id: "avatar2",
    name: "동순",
    url: "/avatar2.glb",
    preview: "/avatars/preview2.png",
    description: "밝고 활발한 여자 아바타",
  },
  {
    id: "avatar3",
    name: "혜미",
    url: "/avatar3.glb",
    preview: "/avatars/preview3.png",
    description: "차분하고 친근한 여자 아바타",
  },
];

interface AvatarState {
  selectedId: string;
  setAvatar: (id: string) => void;
  selectedAvatar: () => AvatarOption;
}

export const useAvatarStore = create<AvatarState>()(
  persist(
    (set, get) => ({
      selectedId: "default",
      setAvatar: (id) => set({ selectedId: id }),
      selectedAvatar: () =>
        AVATAR_OPTIONS.find((a) => a.id === get().selectedId) ?? AVATAR_OPTIONS[0],
    }),
    { name: "fairytale-avatar" }
  )
);
