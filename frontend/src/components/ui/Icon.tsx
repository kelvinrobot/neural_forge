import type { SVGProps } from "react";

export type IconName =
  | "face"
  | "eye"
  | "mouth"
  | "hand"
  | "movement"
  | "posture"
  | "skin"
  | "sun"
  | "moon"
  | "brightness"
  | "blur"
  | "arrowIn"
  | "arrowOut"
  | "rotate"
  | "level"
  | "hands"
  | "body"
  | "camera"
  | "check"
  | "x"
  | "skip"
  | "repeat"
  | "dot"
  | "alert"
  | "shield"
  | "speaker"
  | "speakerOff"
  | "pause"
  | "play"
  | "stop"
  | "timer"
  | "light"
  | "download"
  | "print"
  | "trash"
  | "send"
  | "plus"
  | "chevronRight"
  | "wifiOff"
  | "video"
  | "videoOff"
  | "mic"
  | "micOff"
  | "phoneOff"
  | "search"
  | "copy"
  | "upload"
  | "bell";

const paths: Record<IconName, JSX.Element> = {
  face: (
    <>
      <circle cx="12" cy="12" r="8.5" />
      <circle cx="9" cy="10.5" r="0.9" fill="currentColor" />
      <circle cx="15" cy="10.5" r="0.9" fill="currentColor" />
      <path d="M8.8 14.8c1.8 1.6 4.6 1.6 6.4 0" />
    </>
  ),
  eye: (
    <>
      <path d="M2.5 12s3.5-6 9.5-6 9.5 6 9.5 6-3.5 6-9.5 6-9.5-6-9.5-6Z" />
      <circle cx="12" cy="12" r="2.8" />
    </>
  ),
  mouth: (
    <>
      <path d="M4 11c2.5-2.5 5.5-3 8-1 2.5-2 5.5-1.5 8 1-2.5 4.5-5.2 6-8 6s-5.5-1.5-8-6Z" />
      <path d="M4 11c4 1.5 12 1.5 16 0" />
    </>
  ),
  hand: (
    <path d="M7.5 12V6.5a1.5 1.5 0 0 1 3 0V11m0-5.5V4.5a1.5 1.5 0 0 1 3 0V11m0-5a1.5 1.5 0 0 1 3 0v6m0-3.5a1.5 1.5 0 0 1 3 0V14a7 7 0 0 1-7 7h-.5a6.5 6.5 0 0 1-5.6-3.2L4 13.5a1.6 1.6 0 0 1 2.6-1.8l.9 1.3" />
  ),
  movement: (
    <>
      <path d="M3 12h3l2-5 3.5 10 2.5-7 1.5 2H21" />
    </>
  ),
  posture: (
    <>
      <circle cx="12" cy="4.5" r="2" />
      <path d="M12 7v7m0 0-3 6.5M12 14l3 6.5M7 10.5h10" />
    </>
  ),
  skin: (
    <>
      <rect x="4" y="4" width="16" height="16" rx="4" />
      <circle cx="9.5" cy="10" r="1.6" />
      <circle cx="14.5" cy="14.5" r="1" />
      <circle cx="15" cy="9" r="0.6" fill="currentColor" />
    </>
  ),
  sun: (
    <>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.3 5.3l1.4 1.4M17.3 17.3l1.4 1.4M5.3 18.7l1.4-1.4M17.3 6.7l1.4-1.4" />
    </>
  ),
  moon: <path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5Z" />,
  brightness: (
    <>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 8v8" />
      <path d="M12 3v1.5M12 19.5V21M3 12h1.5M19.5 12H21" />
    </>
  ),
  blur: (
    <>
      <circle cx="12" cy="12" r="7" strokeDasharray="2 2.5" />
      <circle cx="12" cy="12" r="3" />
    </>
  ),
  arrowIn: <path d="M4 4l6 6M10 4.5V10H4.5M20 20l-6-6M14 19.5V14h5.5M20 4l-6 6M14 4.5V10h5.5M4 20l6-6M10 19.5V14H4.5" />,
  arrowOut: <path d="M10 10 4 4M4 9.5V4h5.5M14 14l6 6M20 14.5V20h-5.5M14 10l6-6M14.5 4H20v5.5M10 14l-6 6M4 14.5V20h5.5" />,
  rotate: (
    <>
      <path d="M4.5 12a7.5 7.5 0 0 1 13-5.1L19.5 9" />
      <path d="M19.5 4v5h-5" />
      <path d="M19.5 12a7.5 7.5 0 0 1-13 5.1L4.5 15" />
    </>
  ),
  level: (
    <>
      <rect x="3" y="9" width="18" height="6" rx="3" />
      <path d="M10.5 12h3" />
    </>
  ),
  hands: (
    <>
      <path d="M5 13V8a1.3 1.3 0 0 1 2.6 0v4M7.6 8V6.5a1.3 1.3 0 0 1 2.6 0V12M3 12.5l2 4a4 4 0 0 0 3.5 2h.5a3 3 0 0 0 3-3V11" />
      <path d="M19 13V8a1.3 1.3 0 0 0-2.6 0v4M16.4 8V6.5a1.3 1.3 0 0 0-2.6 0V12M21 12.5l-2 4a4 4 0 0 1-3.5 2H15a3 3 0 0 1-3-3" />
    </>
  ),
  body: (
    <>
      <circle cx="12" cy="4" r="2" />
      <path d="M12 6.5v7.5M12 14l-2.5 7M12 14l2.5 7M7 8.5l5 1.5 5-1.5" />
    </>
  ),
  camera: (
    <>
      <path d="M4 8h3l1.5-2h7L17 8h3v11H4z" />
      <circle cx="12" cy="13" r="3.5" />
    </>
  ),
  check: <path d="M5 12.5l4.5 4.5L19 7.5" />,
  x: <path d="M6 6l12 12M18 6 6 18" />,
  skip: <path d="M5 6l8 6-8 6V6ZM17 6v12" />,
  repeat: (
    <>
      <path d="M4 11V9a3 3 0 0 1 3-3h12l-3-3M20 13v2a3 3 0 0 1-3 3H5l3 3" />
    </>
  ),
  dot: <circle cx="12" cy="12" r="4" />,
  alert: (
    <>
      <path d="M12 3.5 2.5 20h19L12 3.5Z" />
      <path d="M12 10v4.5M12 17.2v.1" />
    </>
  ),
  shield: (
    <>
      <path d="M12 3 4.5 6v6c0 4.5 3.2 7.7 7.5 9 4.3-1.3 7.5-4.5 7.5-9V6L12 3Z" />
      <path d="M12 8.5v4.5M12 15.8v.1" />
    </>
  ),
  speaker: (
    <>
      <path d="M4 9.5h3.5L12 5.5v13l-4.5-4H4z" />
      <path d="M15.5 9a4.5 4.5 0 0 1 0 6M18 6.5a8 8 0 0 1 0 11" />
    </>
  ),
  speakerOff: (
    <>
      <path d="M4 9.5h3.5L12 5.5v13l-4.5-4H4z" />
      <path d="M16 9.5l5 5M21 9.5l-5 5" />
    </>
  ),
  pause: <path d="M8.5 5.5v13M15.5 5.5v13" />,
  play: <path d="M7 5l12 7-12 7V5Z" />,
  stop: <rect x="6" y="6" width="12" height="12" rx="2" />,
  timer: (
    <>
      <circle cx="12" cy="13.5" r="7.5" />
      <path d="M12 13.5V9.5M9.5 2.5h5" />
    </>
  ),
  light: (
    <>
      <path d="M9 18h6M10 21h4M12 3a6 6 0 0 0-3.5 10.9c.6.5 1 1.2 1 2V16h5v-.1c0-.8.4-1.5 1-2A6 6 0 0 0 12 3Z" />
    </>
  ),
  download: <path d="M12 4v11m0 0-4.5-4.5M12 15l4.5-4.5M4.5 19.5h15" />,
  print: (
    <>
      <path d="M7 8V3.5h10V8M7 16.5H4.5V8h15v8.5H17" />
      <rect x="7" y="13" width="10" height="7.5" rx="1" />
    </>
  ),
  trash: <path d="M4.5 6.5h15M9.5 6.5V4h5v2.5M6.5 6.5l1 13.5h9l1-13.5M10 10v6.5M14 10v6.5" />,
  send: <path d="M4 12 20 4l-5 16-3.5-6.5L4 12Zm7.5 1.5L20 4" />,
  plus: <path d="M12 5v14M5 12h14" />,
  chevronRight: <path d="m9.5 6 6 6-6 6" />,
  wifiOff: (
    <>
      <path d="M3 3l18 18M8.5 16a5 5 0 0 1 7 0M5 12.5a10 10 0 0 1 4-2.4M14.5 10a10 10 0 0 1 4.5 2.5M2 9a15 15 0 0 1 5-3M11 5.1A15 15 0 0 1 22 9" />
      <circle cx="12" cy="19.5" r="0.6" fill="currentColor" />
    </>
  ),
  video: (
    <>
      <rect x="3" y="6.5" width="12.5" height="11" rx="2" />
      <path d="m15.5 10.5 5.5-3v9l-5.5-3" />
    </>
  ),
  videoOff: (
    <>
      <path d="M3 3l18 18M15.5 11.5v-3a2 2 0 0 0-2-2H8.5M5 6.6A2 2 0 0 0 3 8.5v7a2 2 0 0 0 2 2h8.5a2 2 0 0 0 1.6-.8M15.5 10.5l5.5-3v9" />
    </>
  ),
  mic: (
    <>
      <rect x="9" y="3" width="6" height="11" rx="3" />
      <path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21" />
    </>
  ),
  micOff: (
    <>
      <path d="M3 3l18 18M9 9v2a3 3 0 0 0 5.1 2.1M15 9.5V6a3 3 0 0 0-5.7-1.3M5.5 11a6.5 6.5 0 0 0 10.4 5.2M18.5 11a6.4 6.4 0 0 1-.6 2.7M12 17.5V21" />
    </>
  ),
  phoneOff: (
    <path d="M3.5 14.5c4.7-4.7 12.3-4.7 17 0l-2.3 2.3-3.2-1.6v-2.4a10 10 0 0 0-6 0v2.4l-3.2 1.6-2.3-2.3Z" />
  ),
  search: (
    <>
      <circle cx="11" cy="11" r="6.5" />
      <path d="m16 16 4.5 4.5" />
    </>
  ),
  copy: (
    <>
      <rect x="8.5" y="8.5" width="11" height="11" rx="2" />
      <path d="M15.5 8.5V6.5a2 2 0 0 0-2-2h-7a2 2 0 0 0-2 2v7a2 2 0 0 0 2 2h2" />
    </>
  ),
  upload: <path d="M12 15.5V4.5m0 0L7.5 9M12 4.5 16.5 9M4.5 19.5h15" />,
  bell: (
    <>
      <path d="M6 16.5V11a6 6 0 0 1 12 0v5.5l1.5 1.5h-15L6 16.5Z" />
      <path d="M10 20.5a2 2 0 0 0 4 0" />
    </>
  ),
};

interface IconProps extends SVGProps<SVGSVGElement> {
  name: IconName;
  size?: number;
}

export function Icon({ name, size = 20, ...rest }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.7}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...rest}
    >
      {paths[name]}
    </svg>
  );
}
