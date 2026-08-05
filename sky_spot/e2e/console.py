"""Console output module for E2E testing.

This module provides the SimulationConsole class for clean, structured
console output during E2E simulations.
"""

import time
from rich.console import Console

# Console for clean streaming output
console = Console()


class SimulationConsole:
    """Clean streaming console output for simulation events."""

    def __init__(self):
        self.last_cost = 0.0
        self.last_active_region = None
        self.start_time = time.time()
        self.task_duration_hours = 0.0
        self.deadline_hours = 0.0

    def _ts(self) -> str:
        """Current timestamp string."""
        return time.strftime("%H:%M:%S")

    def _elapsed(self) -> str:
        """Elapsed time since start."""
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        mins = int((elapsed % 3600) // 60)
        return f"{hours}h{mins:02d}m"

    def start(
        self,
        strategy_name: str,
        regions: list,
        task_duration_h: float,
        deadline_h: float,
        task_id: str = "",
    ):
        """Print simulation start banner."""
        console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
        console.print(f"[bold]🚀 Starting[/bold]")
        if task_id:
            console.print(f"   ID: [magenta]{task_id}[/magenta]")
        console.print(f"   Strategy: [green]{strategy_name}[/green]")
        console.print(
            f"   Zones: [yellow]{len(regions)}[/yellow] zones across {len(set(r[:-1] for r in regions))} regions"
        )
        console.print(
            f"   Task: [cyan]{task_duration_h:.1f}h[/cyan] | Deadline: [cyan]{deadline_h:.1f}h[/cyan]"
        )
        console.print(f"[bold cyan]{'=' * 60}[/bold cyan]\n")
        self.start_time = time.time()
        self.task_duration_hours = task_duration_h
        self.deadline_hours = deadline_h

    def tick(
        self,
        tick_num: int,
        progress_pct: float,
        cost: float,
        active_instances_with_age: dict,
        region_names: list,
        gap_seconds: float,
    ):
        """Print tick summary every tick."""
        # Calculate task progress: done and remaining
        done_hours = (progress_pct / 100.0) * self.task_duration_hours
        remaining_task_hours = self.task_duration_hours - done_hours
        done_h = int(done_hours)
        done_m = int((done_hours - done_h) * 60)
        remain_h = int(remaining_task_hours)
        remain_m = int((remaining_task_hours - remain_h) * 60)
        # Calculate remaining time until deadline
        elapsed_sec = time.time() - self.start_time
        deadline_sec = self.deadline_hours * 3600
        deadline_remaining_sec = max(0, deadline_sec - elapsed_sec)
        dl_h = int(deadline_remaining_sec // 3600)
        dl_m = int((deadline_remaining_sec % 3600) // 60)

        # Build status string: what's currently running with age
        if active_instances_with_age:
            running_parts = []
            for region_idx, (
                cluster_type,
                age_ticks,
            ) in active_instances_with_age.items():
                region_name = (
                    region_names[region_idx]
                    if region_idx < len(region_names)
                    else f"R{region_idx}"
                )
                icon = "🟢" if cluster_type.name == "SPOT" else "🔵"
                # Convert age_ticks to hours/minutes
                age_sec = age_ticks * gap_seconds
                age_h = int(age_sec // 3600)
                age_m = int((age_sec % 3600) // 60)
                running_parts.append(f"{icon}{region_name} {age_h}h{age_m:02d}m")
            status = " ".join(running_parts)
        else:
            status = "[dim]idle[/dim]"

        console.print(
            f"[dim]{self._ts()}[/dim] {status} | "
            f"[green]{done_h}h{done_m:02d}m[/green]/[yellow]{remain_h}h{remain_m:02d}m[/yellow] "
            f"[bold green]{progress_pct:.0f}%[/bold green] | "
            f"[yellow]${cost:.2f}[/yellow] | {self._elapsed()}/[cyan]{dl_h}h{dl_m:02d}m[/cyan]"
        )

    def launch(
        self, region_idx: int, region_name: str, cluster_type: str, is_spot: bool
    ):
        """Print launch event."""
        icon = "🟢" if is_spot else "🔵"
        type_str = "[green]SPOT[/green]" if is_spot else "[blue]ON_DEMAND[/blue]"
        console.print(
            f"[dim]{self._ts()}[/dim] {icon} Launched {type_str} in [cyan]{region_name}[/cyan]"
        )
        self.last_active_region = region_name

    def terminate(self, region_idx: int, region_name: str, cluster_type: str):
        """Print terminate event."""
        console.print(
            f"[dim]{self._ts()}[/dim] ⬛ Terminated [yellow]{cluster_type}[/yellow] in [cyan]{region_name}[/cyan]"
        )

    def preemption(self, region_idx: int, region_name: str, event_type: str = "spot"):
        """Print preemption event."""
        if event_type == "job_failed":
            console.print(
                f"[dim]{self._ts()}[/dim] [bold orange1]💥 JOB FAILED[/bold orange1] in [cyan]{region_name}[/cyan] "
                f"[dim](watchdog timeout or NCCL error)[/dim]"
            )
        elif event_type == "ondemand_failure":
            console.print(
                f"[dim]{self._ts()}[/dim] [bold yellow]⚠️  ON_DEMAND FAILED[/bold yellow] in [cyan]{region_name}[/cyan]"
            )
        else:
            console.print(
                f"[dim]{self._ts()}[/dim] [bold red]⚠️  PREEMPTED[/bold red] in [cyan]{region_name}[/cyan]"
            )

    def migration(self, src_region: str, dst_region: str, size_gb: float, is_recovery: bool = False):
        """Print migration start event.

        Args:
            is_recovery: True if this is recovery after preemption, False if proactive migration
        """
        if is_recovery:
            # Recovery after preemption - show preemption first, then recovery
            console.print(
                f"[dim]{self._ts()}[/dim] [bold red]⚠️  PREEMPTED[/bold red] in [cyan]{src_region}[/cyan]"
            )
            console.print(
                f"[dim]{self._ts()}[/dim] 🔄 Recovering [cyan]{src_region}[/cyan] → [cyan]{dst_region}[/cyan] "
                f"({size_gb:.1f} GB)"
            )
        else:
            # Proactive migration - strategy decided to move
            console.print(
                f"[dim]{self._ts()}[/dim] 📦 Migrating [cyan]{src_region}[/cyan] → [cyan]{dst_region}[/cyan] "
                f"({size_gb:.1f} GB) [dim](strategy decision)[/dim]"
            )

    def migration_complete(
        self,
        src_region: str,
        dst_region: str,
        size_gb: float,
        elapsed_s: float,
        speed_gbps: float,
        skipped: int = 0,
        is_recovery: bool = False,
    ):
        """Print migration completion."""
        skip_info = f" [dim](skipped {skipped} identical)[/dim]" if skipped > 0 else ""
        action = "Recovery" if is_recovery else "Migration"
        console.print(
            f"[dim]{self._ts()}[/dim] [green]✓[/green] {action} complete: "
            f"[cyan]{src_region}[/cyan] → [cyan]{dst_region}[/cyan] | "
            f"[bold]{size_gb:.1f} GB[/bold] in [bold]{elapsed_s:.1f}s[/bold] "
            f"([green]{speed_gbps:.2f} Gbps[/green]){skip_info}"
        )

    def safety_net(self, region_name: str):
        """Print safety net trigger."""
        console.print(
            f"[dim]{self._ts()}[/dim] [bold yellow]🛡️  SAFETY NET[/bold yellow] triggered → [cyan]{region_name}[/cyan]"
        )

    def done(
        self,
        total_cost: float,
        migrations: int,
        preemptions: int,
        final_progress: float,
        compute_cost: float = 0.0,
        transfer_cost: float = 0.0,
    ):
        """Print completion summary."""
        console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
        console.print(
            f"[bold]✅ Complete[/bold] | Elapsed: {self._elapsed()}"
        )
        console.print(f"   Final Progress: [green]{final_progress:.1f}%[/green]")
        # Show cost breakdown if transfer cost is non-zero
        if transfer_cost > 0.01:
            console.print(
                f"   Total Cost: [yellow]${total_cost:.2f}[/yellow] "
                f"[dim](Compute: ${compute_cost:.2f} + Transfer: ${transfer_cost:.2f})[/dim]"
            )
        else:
            console.print(f"   Total Cost: [yellow]${total_cost:.2f}[/yellow]")
        console.print(
            f"   Migrations: [cyan]{migrations}[/cyan] | Preemptions: [red]{preemptions}[/red]"
        )
        console.print(f"[bold cyan]{'=' * 60}[/bold cyan]\n")

    def error(self, msg: str):
        """Print error message."""
        console.print(f"[dim]{self._ts()}[/dim] [bold red]❌ ERROR:[/bold red] {msg}")
