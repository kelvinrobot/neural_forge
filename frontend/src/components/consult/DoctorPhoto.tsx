import type { Doctor } from "../../data/doctors";

export function DoctorPhoto({ doctor, className = "w-full" }: { doctor: Doctor; className?: string }) {
  return (
    <img
      src={encodeURI(doctor.photo)}
      alt={`Photo of ${doctor.name}`}
      loading="lazy"
      className={`aspect-[4/5] shrink-0 rounded-[8px] bg-ink-100 object-cover object-top dark:bg-ink-800 ${className}`}
    />
  );
}
